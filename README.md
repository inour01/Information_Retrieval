# Information_Retrieval

Instructions to Run Programs:

Place the python file in a folder which should contain the following items:
Folder with “content/AP_collection/coll/” which contains all the documents in the corpus
stopwords.txt, which contains all stopwords
queries.txt , which contains all 50 queries
Please ensure you have NLTK downloaded and imported to use the module punkt
Then in your IDE of choice run the python program, it will produce two documents VocabResults.txt, which contains a sample of 100 vocab words and the size of the entire vocabulary. Along with Results.txt which contains the result output of the querying.




Step 1 – Preprocessing:
Preprocess is a function defined in our code that accepts a single document as input. The function extracts the text from the input document in between the TEXT and TEXT tags using regular expressions. The nltk.word tokenize() function is then used to tokenize the extracted text. The non-alphabetic tokens are eliminated, and the remaining tokens are all changed to lowercase. The code then reads the file "stopwords.txt" in read-only mode, splits the file's lines to produce a set of custom stopwords, and then deletes tokens from the list of tokens that are in the set of custom stopwords. The function returns the list of stemmed tokens after creating an instance of the Porter Stemmer class and applying it to all remaining tokens to stem them.


Step 2 – Indexing:
Parameters
The function buildIndex handles building up the entire inverted index from the corpus of preprocessed documents. It takes the variable preprocessed which is an array where each entry is an individual preprocessed document and the variable doc_number_tokens which is an array that maps the Document IDs to the Document names.
Data Structures
The main data structures we use for holding information in the index are Dictionaries (Hash Tables). The benefits of python dictionaries are that they allow for quick access and utilizing a specific library allow for nested dictionary entries. The variable mainIndex will hold the inverted Index, where the keys are terms and the values follow this format (example uses index[term]:
mainIndex[term][0]= IDF Value for the term
mainIndex[1….x] = An array containing the important values for each relevant document to the term, formatted like: [Document Name, TF-IDF Score, TF Score, Number of Occurrence in This Specific Document]
Steps
• Place all terms from every document into a set so it filters for unique terms only
• For each document check for the number of occurrences of each unique term, if the term is present update mainIndex[term] with the Document Name, TF Value, and Number of occurrences.
• Iterate through all terms and calculate the IDF value then for each relevant document,
• Iterate through mainIndex and multiple the calculated IDF to the TF value for each relevant term in a document and append it to the document details array for the relevant documents
Returns
This function returns the mainIndex and weightsArray


Step 3 – Retrieval and Ranking:
Extracting the Queries
The first step was to extract all the queries. We decided to use both query title and query description. However, we could easily edit the function so that it retrieves only the query titles. queriesArray(fileLocation) is a function that takes a file location (the queries.txt file) and returns an array
of queries with their query number. This function also removes punctuation and common expressions like “The document will”.
Function output: [[query number, query], [query number, query], …]
Retrieval and Ranking – Dictionaries and Arrays
The next step was to use the inverted index from the Step 2 to find the set of documents that contain at least one of the query words. Get the query and document weights for the cosine similarity computations. Then we would be able to rank the relevance of each document. retrieval_and_ranking(query, invertedIndex, weightsArray) is a function that takes a query, an inverted index and an array as input. The invertedIndex is the one created in step 2 and the array is an array containing the term weights. In this function, the related documents are determined and stored in a dictionary, the query and document weights are calculated and also stored in dictionaries / arrays, the cosine similarity scores are evaluated using a helper function (cosine_sim), and finally returns a dictionary of documents and their score, sorted in descending order.
Function output: {dict name : score, dict name : score, ...}
Steps for Retrieval and Ranking Function
• Loop through each word in the query to the word frequency as well as the related documents
• Get the maximum frequency in the query
• For each word in the query word frequency dictionary, calculate the query word weight and add it to a query weight dictionary
• Iterate through each relevant document – get the document word weight for each word in the query and append it to a docWeights array. After the doc weights are calculated for a document, compute the cosine similarity score (using the cosine_sim helper function), and add it to a cosimScores dictionary
• Sort the dictionary by values in descending order and return the final dictionary
