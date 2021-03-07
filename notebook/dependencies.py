import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords');
import time
from collections import defaultdict
from array import array
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
from config import *
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
import string
import unidecode
from wordcloud import WordCloud, ImageColorGenerator
import re
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()
import numpy as np
import matplotlib.pyplot as plt

 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from nltk import bigrams
import time
from nltk.util import ngrams

def remove_links(text):
    if text:
        return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    
    #In case there is no text
    return ""


def remove_hashtags(text):
    if text:
        return re.sub(r'#\w+ ?', '', text)
    
    return ""
def remove_accents(text):
    if text:
        return unidecode.unidecode(text)
        

    # In case there is no text
    return ""

def remove_punctuation_marks(text):
    if text:
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        new_text = text.translate(translator)
        return " ".join(new_text.split())
        
    # In case there is no text
    return ""

def text_to_lower_case(text):
    if text:
        return text.lower()
    
    # In case there is no text
    return ""

def remove_emojis(text):
    if text:
        # TODO: Remove emojis (tip: search for encode - decode)
        returnString = ""
        for character in text:
            try:
                character.encode("ascii")
                returnString += character
            except UnicodeEncodeError:
                returnString += ''
        text = " ".join(returnString.split())
        
    return text

    # In case there is no text
    return ""

def remove_multiple_whitespaces(text):
    if text:
        return " ".join(text.split())

    # In case there is no text
    return ""

def remove_text_marks(text):
    if text:
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # TODO: replace *, ?, ... by spaces
        
        return text.strip()
    
    # In case there is no text
    return ""

def split_text_and_numbers(text):
    temp = re.compile("([a-zA-Z]+)([0-9]+)") 
    try:
        res = temp.match(text).groups()
        text = " ".join(res)
    except:
        text = text
    return text

def remove_alone_numbers(text):
    if text:
        text = ' '.join(filter(lambda word:word.replace('.','').isdigit()==False, text.split()))
        
        return text
    
    return ""

def remove_stopwords(text):
    text = text.split()
    return " ".join([x for x in text if x not in STOPWORDS])

def stemming(text):
    text = text.split()
    return " ".join([ps.stem(x) for x in text])

def clean_text(text):
    # Apply the different functions in order to clean the text
    text = remove_links(text)
    text = remove_hashtags(text)
    text = text_to_lower_case(text)
    text = remove_text_marks(text)
    text = remove_accents(text)
    text = remove_emojis(text)
    text = split_text_and_numbers(text)
    text = remove_alone_numbers(text)
    text = remove_multiple_whitespaces(text)
    text = remove_punctuation_marks(text)
    text = remove_stopwords(text)
    text = stemming(text)
    
    # Return
    return text.split()

def load_tweets():
    with open("one.json", "rb") as f:
        lines = f.readlines()
        lines = [json.loads(str_) for str_ in lines]
    df_tweets = pd.DataFrame(lines)
    df_tweets = df_tweets.loc[~df_tweets['text'].str.startswith("RT")]
    df_tweets.reset_index(drop = True, inplace=True)
    df_tweets = df_tweets.drop_duplicates(subset='text', keep="last")
    df_tweets.reset_index(drop = True, inplace=True)
    df_tweets['id_tweet'] = df_tweets.index
    df_tweets['join'] = df_tweets[['id_tweet', 'text']].apply(lambda x: '{}|{}'.format(x[0],x[1]), axis=1)
    return df_tweets

def create_index_tfidf(lines, numDocuments):

        
    index=defaultdict(list)
    tf=defaultdict(list) #term frequencies of terms in documents (documents in the same order as in the main index)
    df=defaultdict(int)         #document frequencies of terms in the corpus
    titleIndex=defaultdict(str)
    idf=defaultdict(float)
    
    for line in lines: # Remember, lines contain all documents, each line is a document
        line_arr = line.split("|")
        page_id = int(line_arr[0])
        terms = clean_text(line_arr[1])         
        
        termdictPage={}

        for position, term in enumerate(terms): ## terms contains page_title + page_text
            try:
                # if the term is already in the dict append the position to the corrisponding list
                termdictPage[term][1].append(position)  
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                termdictPage[term]=[page_id, array('I',[position])] #'I' indicates unsigned int (int in python)
        
        #normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm=0
        for term, posting in termdictPage.items(): 
            # posting is a list containing doc_id and the list of positions for current term in current document: 
            # posting ==> [currentdoc, [list of positions]] 
            # you can use it to inferr the frequency of current term.
            norm+=len(posting[1])**2
        norm=math.sqrt(norm)


        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in termdictPage.items():     
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4))  ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term]= df[term] + 1  # increment df for current term
        
        #merge the current page index with the main index
        for termpage, postingpage in termdictPage.items():
            index[termpage].append(postingpage)
            
            # Compute idf following the formula (3) above. HINT: use np.log
    for term in df:
        idf[term] = np.round(np.log(float(numDocuments/df[term])),4)
            
    return index, tf, df, idf, titleIndex

def create_index(df_tweets):
    start_time = time.time()
    numDocuments = len(df_tweets)
    index, tf, df, idf, titleIndex= create_index_tfidf(df_tweets, numDocuments)
    end_time = time.time()
    print("Total time to create the index: {} seconds" .format(np.round(end_time - start_time,2)))
    return index, tf, df, idf, titleIndex

def rankDocuments(terms, docs, index, idf, tf, titleIndex):

        
    # I'm interested only on the element of the docVector corresponding to the query terms 
    # The remaing elements would became 0 when multiplied to the queryVector
    docVectors=defaultdict(lambda: [0]*len(terms)) # I call docVectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    queryVector=[0]*len(terms)    

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms) # get the frequency of each term in the query. 
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    # HINT: use when computing tf for queryVector
    
    query_norm = la.norm(list(query_terms_count.values()))
    
    for termIndex, term in enumerate(terms): #termIndex is the index of the term in the query
        if term not in index:
            continue
                    
        ## Compute tf*idf(normalize tf as done with documents)
        queryVector[termIndex]=query_terms_count[termIndex]/query_norm * idf[term] 

        # Generate docVectors for matching docs
        for docIndex, (doc, postings) in enumerate(index[term]):
            # Example of [docIndex, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....
            
            #tf[term][0] will contain the tf of the term "term" in the doc 26            
            if doc in docs:
                docVectors[doc][termIndex]=tf[term][docIndex] * idf[term]  # TODO: check if multiply for idf

    # calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine siilarity
    # see np.dot
    
    docScores=[ [np.dot(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items() ]
    docScores.sort(reverse=True)
    resultDocs=[x[1] for x in docScores]
    #print document titles instead if document id's
    #resultDocs=[ titleIndex[x] for x in resultDocs ]
    if len(resultDocs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)    
    #print ('\n'.join(resultDocs), '\n')
    return resultDocs

def search_tf_idf(query, index, tf, df, idf, titleIndex, data):
    '''
    output is the list of documents that contain any of the query terms. 
    So, we will get the list of documents for each query term, and take the union of them.
    '''
    query=clean_text(query)
    #docs=set()
    docs = set(np.linspace(0,len(data)-1, len(data)))
    for term in query:
    ## START DODE
        try:
            # store in termDocs the ids of the docs that contain "term"                        
            termDocs=[posting[0] for posting in index[term]]
            # docs = docs Union termDocs
            docs = docs.intersection(termDocs)
        except:
            #term is not in index
            pass
    docs=list(docs)
    ranked_docs = rankDocuments(query, docs, index, idf, tf, titleIndex)   
    return ranked_docs

def google(query, index, tf, df, idf, titleIndex, data, top = 20):
    print('Query: {}'.format(query))

    ranked_docs = search_tf_idf(query, index, tf, df, idf, titleIndex, data)    
    

    results = pd.DataFrame(columns=['tweet', 'username', 'date', 'hashtags', 'likes', 'retweets', 'url'])

    count = 0
    for d_id in ranked_docs[:top] :
        results.loc[count,'tweet'] = data.loc[d_id, "text"]
        results.loc[count,'username'] = data.loc[d_id, "user"]['screen_name']
        results.loc[count,'date'] = data.loc[d_id, "created_at"]
        
        hashtags = data['entities'][d_id]['hashtags']
        aux = []
        for h in hashtags:
            aux.append(h['text'])
        
        #results.loc[count,'hashtags'] = [data.loc[d_id,'entities']['hashtags'][i]['text'] for i in range(len(data.loc[0,'entities']['hashtags']))]
        results.loc[count,'hashtags'] = aux
        results.loc[count,'likes'] = data.loc[d_id, "favorite_count"]
        results.loc[count,'retweets'] = data.loc[d_id, "retweet_count"]
        results.loc[count,'url'] = "https://twitter.com/twitter/statuses/"+str(data.loc[d_id, "id"])
        try:
            results.loc[count,'cluster'] = data.loc[d_id, "cluster"]
        except:
            print('An exception ocurred')
        count +=1
    return results