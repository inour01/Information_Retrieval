# -*- coding: utf-8 -*-

#Run this once to install necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import re
import math
import pandas as pd
import numpy as np
import fileinput
import sys

def preprocess(document):  # defines a new function called preprocess that takes in a single document
    # as an input.

    #  This line  uses regular expressions to extract the text between the <TEXT> and </TEXT> tags.
    # document = re.search("<TEXT>(.*?)</TEXT>", document, flags=re.DOTALL).group(1)
    #  This line tokenizes the text using the nltk.word_tokenize() function.
    tokens = nltk.word_tokenize(document)
    #  This line removes any tokens that are not alphabetic.
    tokens = [token for token in tokens if token.isalpha()]
    #  This line converts all remaining tokens to lowercase.
    tokens = [token.lower() for token in tokens]
    # This line opens a file named 'stopwords.txt' in read mode and assigns the file object 
    #to the variable 'f'.
    #with open('/content/stopwords.txt', "r") as f:
    with open('content/stopwords.txt', "r") as f:
    # This line creates a set of custom stopwords by splitting the lines in the file 
    #and assigns it to the variable 'custom_stopwords'.
      custom_stopwords = set(f.read().splitlines())
    # This line removes tokens from the list 'tokens' 
    #that are in the set of custom stopwords.
    tokens = [token for token in tokens if token not in custom_stopwords]
    # This line creates an instance of the PorterStemmer class, which will be used to stem the remaining tokens.
    stemmer = PorterStemmer()
    # This line applies the stemmer to all remaining tokens and returns the resulting list of stemmed tokens.
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def buildIndex(preprocessed, doc_number_tokens): #defines a new function called buildIndex which will create the inverted index based on the inf-df structure
    from collections import defaultdict
    mainIndex = defaultdict(dict)
    #termFrequencyAllDocs = defaultdict(dict)
    indexTerms = set()

    #Place all terms from every document in a set so that we can filter for unique terms
    for myList in preprocessed:
      for item in myList:
        indexTerms.add(item)


    #Initialize Dictionaries by setting default values
    for term in indexTerms:
      mainIndex.setdefault(term, [])
    
    print("Total Number of Terms: ", len(indexTerms))
    print("Total Number of Documents: ", len(preprocessed))

    currentDocID = 0 #counter
    #Iterates through all documents one!
    for doc in preprocessed:
      documentLength = len(doc)
      for term in indexTerms:
        occurances = doc.count(term) #tracks the number of occurances within the text
        if (occurances > 0):
          tf = occurances/documentLength
          (mainIndex[term]).append([doc_number_tokens[currentDocID], tf, occurances])
      currentDocID = currentDocID + 1

    for term in indexTerms: #iterate through all the documents a term is in for each document and calculate IDF and TF-IDF
      idf_equation = len(preprocessed)/(len(mainIndex[term])) #Minus 1 bc the first entry in the array accounts for the total occurances of the term across the corpus
      idf = math.log(idf_equation, 2)
      (mainIndex[term]).insert(0,idf)
      term_array_length = len(mainIndex[term])
      for i in range(1,term_array_length):
        tf = mainIndex[term][i][1]
        tf_idf = idf*tf
        mainIndex[term][i].insert(1, tf_idf)

    return mainIndex

#


def queriesArray(fileLocation): #defines a new function to return an array of all queries [[query number, query title]]
  queries = []
  qNum = 1
  start = '<title>'
  end = '<narr>'
  pattern = r'{}.*?{}'.format(re.escape(start), re.escape(end)) # Escape special chars, build pattern dynamically
  with open(fileLocation) as queriesFile:
      contents = queriesFile.read()                     # Read file into a variable
      for query in re.findall(pattern, contents, re.S): 
        query = query.split(start)[1].split(end)[0].strip().replace("\n", "")
        query = query.replace(",", "")
        query = query.replace(":", "")
        query = query.replace(".", "")
        query = query.replace("/", " ")
        query = query.replace("<desc>", "")
        query = query.replace("(", "")
        query = query.replace(")", "")
        query = query.replace("Document will", "")
        query = query.replace("The document will", "")
        query = query.replace("A relevant document", "")
        query = query.lower()
        queries.append([qNum,query])
        qNum += 1
  return queries

def retrieval_and_ranking(query, invertedIndex): # define new function to return the doc ranking scores for a given query (dict of {docName:score})
    shortenedIndex = {}
    relevantDocs = {}
    queryWordFreq = {}
    queryWeights = {}
    cosimScores = {}
    
    # Loop through each word in the query to get query word tf for words in the index + all relevant documents
    for qword in query.split():
        if invertedIndex[qword]:
            containsWord = invertedIndex[qword]
            if qword not in shortenedIndex:
              shortenedIndex[qword] = invertedIndex[qword]
            for doc in range(1,len(containsWord)):
              docName = invertedIndex[qword][doc][0]
              # go through array to add related docs
              if docName not in relevantDocs:
                relevantDocs[docName] = 0
            # populate queryWordFreq
            if qword not in queryWordFreq:
              queryWordFreq[qword] = 1
            else:
              queryWordFreq[qword] += 1
   
   #Get maximum frequency in the query
    if len(list(queryWordFreq.values())) > 0:
      maxqueryfrequency = max(list(queryWordFreq.values()))
    else:
      maxqueryfrequency = 1

    # Loop through each word in queryWordFreq to populate queryWeights
    for qword in queryWordFreq:
        if invertedIndex[qword]:
          #maxtf = invertedIndex[qword][0]
          idf = invertedIndex[qword][0]
          qWordWeight = (queryWordFreq[qword]/maxqueryfrequency)*idf
          queryWeights[qword] = qWordWeight

    # iterates through all relevant docs and 
    for doc in relevantDocs: 
      docWeights = []
      for qword in queryWeights.keys():
        docWeightsNames = []
        for docNum in range(1, len(invertedIndex[qword])):
          docName = invertedIndex[qword][docNum][0]
          docWeightsNames.append(docName)
          if doc == docName:
            tfidf = invertedIndex[qword][docNum][1]
            docWeights.append(tfidf)
        if doc not in docWeightsNames:
          docWeights.append(0)
      cosim = cosine_sim(list(queryWeights.values()),docWeights)
      cosimScores[doc] = cosim

      # sorted
    cosimScoresSorted = sorted(cosimScores.items(), key=lambda x:x[1], reverse=True)
    cosimScoresSorted = dict(cosimScoresSorted)
    return cosimScoresSorted

#Calculates the cosine similarity between two vectors
def cosine_sim(vec1, vec2):
    vec1 = list(vec1)
    vec2 = list(vec2)
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    return dot_prod / (mag_1 * mag_2)

def getVocab(invertedIndex):
  vocab = list(invertedIndex.keys())
  fp = open('VocabResults.txt', 'w')
  fp.write("Vocab Size: ")
  fp.write(str(len(vocab)))
  fp.write("\n")
  fp.write("Sample 100 Vocabulary Words")
  fp.write("\n")
  for i in range(100):
    fp.write(vocab[i])
    fp.write("\n")

def main():
    print("Start Preprocessing")
    #folder_path = '/content/AP_collection/coll'
    folder_path = 'content/AP_collection/coll'
    preprocessed_document = []
    doc_number_tokens = []
    for file in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, file)):
            continue
        with open(os.path.join(folder_path, file), 'r') as f:
            document = f.read()
            allDocNums = re.findall("<DOCNO>(.*?)</DOCNO>", document, re.DOTALL)
            allDocTexts = re.findall("<TEXT>(.*?)</TEXT>", document, re.DOTALL)
            for documents in range(len(allDocNums)): # for documents in range(len(allDocNums)):
              doc_number = allDocNums[documents]
              doc_number_tokens.append(doc_number)
              doc_text = allDocTexts[documents]
              #doc_text_tokens.append(doc_text)
              preprocessed_document.append(preprocess(doc_text))
    print("Start Inverted Index")
    invertedIndex = buildIndex(preprocessed_document, doc_number_tokens)
    print("Start Vocab")
    getVocab(invertedIndex)
    print("Start Queries")
    #queriesFile = "/content/queries.txt"
    queriesFile = "content/queries.txt"
    for line in fileinput.input(queriesFile,inplace=True):
        sys.stdout.write("{} \n".format(line.rstrip()))

    # returns an array of [queryNum, query]
    queries = queriesArray(queriesFile)
    # create new file Results
    fp = open('Results.txt', 'w')
    # retrieval and ranking
    for query in queries:
      rank = 1
      finalRanking = retrieval_and_ranking(query[1], invertedIndex)
      for docName, score in finalRanking.items():
        if (rank > 1000):
          break
        line = str(query[0]) + " Q0 " + docName  + " " + str(rank) + " " + str(score) + " tag"
        fp.write(line)
        fp.write("\n")
        rank += 1
    
    
    # close file
    fp.close()

if __name__ == '__main__':
    main()
