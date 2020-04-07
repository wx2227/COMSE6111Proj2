import requests
import json
import sys
from stanfordnlp.server import CoreNLPClient
from bs4 import BeautifulSoup
import tika
from tika import parser
from collections import defaultdict
from stanfordnlp.server import to_text

# targeted relations
relation = ["per:schools_attended", "per:employee_or_member_of", "per:cities_of_residence", "org:top_members_employees"]

# named entity to extract
patterns = {relation[0]: ['ORGANIZATION', 'PERSON'],
            relation[1]: ['ORGANIZATION', 'PERSON'],
            relation[2]: ['PERSON', ['LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY']],
            relation[3]: ['ORGANIZATION', 'PERSON']}

# transfer the input relation number to targeted relation
toRelation = {1: relation[0], 2: relation[1], 3: relation[2], 4: relation[3]}


# retrieve the top-10 search results from google's api
# @return list of link to html file
def retrieve_links(key, cx, q):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"cx": cx,
              "q": q,
              "key": key,
              "num": 10}

    response = requests.get(url=url, params=params).json()
    data = response["items"]
    
    links = [item['link'] for item in data]
    
    return links


# parse the url with beautifulsoup
# extract the web content of url 
def parse_html_bs(link):
    #parse the url with beautifulsoup
    r = requests.get(url=link)
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    text = ""
    for t in soup.find_all(['title', 'header']):
        text += " ".join(t.text.split()) + "\n"
    for p in soup.find_all(['p', 'dl', 'td', 'tl', 'li']):
        text += " ".join(p.text.split()) + "\n"
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5']):
        text += " ".join(h.text.split()) + "\n"
    return text

# parse the url with Tika 
def parse_html(link):
    try:
        parsed = parser.from_file(link)
        content = parsed.get("content", "")
        strip_content = ' '.join(content.split())
        return strip_content
    except:
        return parse_html_bs(link)


# return the first 20000 characters of extracted web content
def process_link(link):   
    text = parse_html(link)
    print("\tWebpage length (num characters): %d" % len(text))
    
    return text[:20000]


def extract_relation(sentence):
    print("\t\t=== Extracted Relation ===")
    print("\t\tSentence: %s" % sentence)


# extract the named entity of the given content
def pipeline1(text, r, t):
    extractedRelations = []
    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'], timeout=450000, memory='4G', endpoint="http://localhost:9000", threads=7) as pipeline1:
        print("\tAnnotating the webpage using [tokenize, ssplit, pos, lemma, ner] annotators ...")
        ann = pipeline1.annotate(text)
        sentenceNumber = len(ann.sentence)
        namedEntity = patterns[toRelation[r]]
        print("\tExtracted %d sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ..." % sentenceNumber)
        
        # if a sentence has the targeted two named entities
        # add the sentence to the list that the element of which perform extracting kbp annotations
        processedSentence = []
        for i, sentence in enumerate(ann.sentence):
            # check if those named entity in the query all appear in the extract sentence
            firstEntity = False
            secondEntity = False
            for token in sentence.token:
                if toRelation[r] == relation[2]:
                    if token.ner == namedEntity[0]:
                        firstEntity = True
                    if token.ner in namedEntity[1]:
                        secondEntity = True
                else:
                    if token.ner == namedEntity[0]:
                        firstEntity = True
                    if token.ner == namedEntity[1]:
                        secondEntity = True
                        
            # if both targeted named entity appear, the sentence adds to the list
            if firstEntity and secondEntity:
                processedSentence.append([i, to_text(sentence)])
                
        # extract the relations in the list of sentence through pipeline2
        extractedRelations += pipeline2(processedSentence, t)
        print("Extracted kbp annotations for %d out of total %d sentences" % (len(processedSentence), sentenceNumber))
        
    return extractedRelations
    

def pipeline2(processedSentence, t):
    extractedRelations = []
    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner','depparse', 'coref', 'kbp'], timeout=450000, memory='4G', endpoint="http://localhost:9001", threads=7) as pipeline2:
        for item in processedSentence:
            i = item[0]
            text = item[1]
              
            # annorate the text of sentence
            ann = pipeline2.annotate(text)
              
            for sentence in ann.sentence:
                print("\tprocessing %dth sentence." % i)
              
                # extract the kbp annotations of sentence
                for kbpTriple in sentence.kbpTriple:
              
                    # print out the extracted kbp triple whose relation matches with the targeted relation
                    if kbpTriple.relation == toRelation[r]:
                        print("\t\t=== Extracted Relation ===")
                        print("\t\tSentence: %s" % text)
                        print("\t\tConfidence: %f; Subject: %s; Object: %s;" % (kbpTriple.confidence, kbpTriple.subject, kbpTriple.object))
              
                        if kbpTriple.confidence < t:
                            print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
                        elif kbpTriple.confidence >= t:
                            print("\t\tAdding to set of extracted relations.")
                            extractedRelations.append([kbpTriple.confidence, [kbpTriple.subject, kbpTriple.object]])
    return extractedRelations


# remove the relation with same subject and object from the extracted relations
# if their confidence is different, keeps the highest one
def remove_duplicates(extractedRelations):
    tmp = []
    relations = []
    sortedRelations = sorted(extractedRelations, key=lambda x: x[0], reverse=True)
    for item in sortedRelations:
        if item[1] not in tmp:
            tmp.append(item[1])
            relations.append(item)
#             relations.append([item[0]] + item[1])
    
    return relations


# genrate new query based on the cleaned extracted relations
# if the query already be queried before, ignore it
def new_query(sortedRelations, queriedSet):
    tmpQuery = []
    for relation in sortedRelations:
        tmpQuery = list(relation[1][0].split(" ")) + list(relation[1][1].split(" "))
        # the new query should not be queried before
        if " ".join(sorted(tmpQuery)) not in queriedSet:
            break
    return " ".join(tmpQuery)


# the sub process of iterative set expansion
# update the relations for each iterative process
def process(key, cx, q, t, processedUrls, relations):
    links = retrieve_links(key, cx, q)
    for i, link in enumerate(links):
        if link not in processedUrls:
            # add the link to the set of processed urls
            processedUrls.add(link)
              
            print("URL (%d / 10): %s" % (i+1, link))
            print("\tFetching text from url ...")
            processedText = process_link(link)
              
            # extract the relations in the content of current link
            try:
                extractedRelations = pipeline1(processedText, r, t)
            except Exception as e:
                extractedRelations = []
                print(str(e), file=sys.stderr)
            
            lenBefore = len(relations)
              
            # add the relations to the list of relations that have been extracted from former links
            relations += extractedRelations
            
            # remove the duplicate relation (with the same subject and object)
            relations = remove_duplicates(relations)
            print("Relations extracted from this website: %d (Overall: %d)" % (len(relations) - lenBefore, len(relations)))
        else:
            print("\tThis link has been processed")
    
    return relations, processedUrls


def display_result(relations):
    print("================== ALL RELATIONS (%d) =================" % len(relations))
    for relation in relations:
        print("Confidence: %f\t| Subject: %s\t| Object: %s" % (relation[0], relation[1][0], relation[1][1]))

# expand the set iteratively
def iterative_expansion(key, cx, r, t, q, k):  
    times = 0
    relations = []
    query = q
    queriedList = [query.split(" ")]
    queriedSet = set(" ".join(sorted(list(query.split(" ")))))
    processedUrls = set() # keep track of the urls already be processed
              
    while len(relations) < k and query:
        print("=========== Iteration: %d - Query: %s ===========" % (times, query))
        
        # get the relations from current set expanding process
        # add those relations to the relations that should be returned as result
        relation, processedUrls = process(key, cx, query, t, processedUrls, relations)
        relations += relation
              
        # remove the duplicate relation (with the same subject and object)
        relations = remove_duplicates(relations)
        
        # display the extracted relations
        display_result(relations)
        
        # if the extracted relations smaller than the targeted extracted number
        # go through the method process with new query
        if len(relations) < k:
            query = new_query(relations, queriedSet)
            queriedList.append(query.split(" "))
            queriedSet.add(" ".join(sorted(query.split(" "))))
            # add up the iterative times
            times += 1
        else:
              break
    
    
if __name__ == "__main__":
    # cx = "011931726167723972512:orkup7yeals"
    # key = "AIzaSyAg_FedCkdEHFmYwRdkqS5Im2zeOjlrC4Y"
    # query = "bill gates microsoft"
    key, cx, r, t, q, k = (sys.argv[x] for x in range(1, 7))
    r = int(r)
    t = float(t)
    k = int(k)
    
    iterative_expansion(key, cx, r, t, q, k)
    
    
    