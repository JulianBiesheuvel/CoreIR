import pyterrier as pt
import math

# Search for the requested documents in the corpus
def search_with_doc_id(doc_id):
    files = open("./collection.tsv", "r", encoding="utf8")


    for file in files:
        docno, passage = file.split("\t")
        if int(docno) == doc_id:
            files.close()
            return passage
    
    files.close()

# Given a list of document ids, return the documents
def search_multiple_doc_ids(list_docs):
    data = []
    for doc in list_docs:
        data.append(search_with_doc_id(doc))
    return data

# Get all the scores for all the documents for one particular query'
# Parameter is all the result of get_all_scores()
def get_doc_scores_query(res, query):
    return res[res['query']==query]
    
# Get all the scores for all the documents and all the queries in the data collection
def get_all_scores(dataset, bm25):
    topics = dataset.get_topics("test-2019")

    return bm25.transform(topics)

# Return metric for each query ,we can figure out the least or best performing query
def eval_all_queries(metric, res, qrels):
    return pt.Utils.evaluate(res,qrels,metrics = [metric], perquery = True)

# Given a queryId and evaluation metric, return the score for that query and metric
# queryId should be a string
def eval_one_query(queryId, metric, res, qrels):
    return eval_all_queries(metric, res, qrels).get(queryId)

############################
###get freq for queries?####
############################
# Term query frequency in the BM25

# What terms occur in the x-th document?
def get_term_freqs(index, docId):
    di = index.getDirectIndex()
    doi = index.getDocumentIndex()
    lex = index.getLexicon()
    docid = docId #docids are 0-based
    
    terms = dict()
    
    #NB: postings will be null if the document is empty
    for posting in di.getPostings(doi.getDocumentEntry(docid)):
        termid = posting.getId()
        lee = lex.getLexiconEntry(termid)
        terms[lee.getKey()] = posting.getFrequency()
        
    return terms

# Given a dictonary of term frequencies and word, return the idf of each term in either the query or the document
def idf(term_freqs):    
    for key in term_freqs:
        val = term_freqs[key]
        term_freqs[key] = [val, round(math.log(8841823/val), 3)]
    return term_freqs