import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import matplotlib.table

ps = PorterStemmer()

def findKN(similarity_vector, k):
    #This function finds k nearest neighbors based on similarity
      
    return np.argsort(-similarity_vector)[:k]

def reviewCh(nearestNeighbors, labels):
    #This function obtains Sentiment  
    positive = 0
    negative = 0
    for neighbor in nearestNeighbors:
        
        
        if int(labels[neighbor]) == 1:
            positive += 1
        else:
            negative += 1
        
    if positive > negative:
        return 1
    else:
        return -1
    
    
# data preprocessing
cls=[]

def preprocess(data):
    #this function remove number and 
    review=[]
    data_len=len(data); 
    for i in range(data_len):
        line = data[i]
        l_len = len(line)
        for t in range(l_len):
            if t ==  1:
                cls.append(line[0:2])
            else:
                if t == 2:
                    text = BeautifulSoup(line[2:l_len], "lxml")
                    letter = re.sub("[^a-zA-Z]", " ", text.get_text())
                    l_case = letter.lower()
                    words = l_case.split()
                    #print words
                    stop = set(stopwords.words("english"))
                    m_words = [wr for wr in words if not wr in stop]
                    
                    text1 = (" ".join(m_words))
                    review.append(text1)  
    return review

def loadData(trainingFile, testingFile):
    #Read the input files and read every line
    
    with open(trainingFile, mode='r', encoding='utf-8') as fr1:
        trainFile = fr1.readlines()
    
    with open(testingFile, mode='r', encoding='utf-8') as fr2:
        testFile = fr2.readlines()
    
    #Split each line in the two files into reviews and labels  
    reviews_train  = [x.split("\t", 1)[1] for x in trainFile]
    sentiments_train = [x.split("\t", 1)[0] for x in trainFile]
    
    #test_sentiments = [x.split("\t", 1)[1] for x in testFile]
    reviews_test = [x.split("\t", 1)[0] for x in testFile]
    
    return reviews_train[1:], reviews_test[1:], sentiments_train[1:]
    


#read input file
reviews_train, reviews_test, sentiments_train = loadData("train_file.txt", "test_file.txt")



# train data processig

review=preprocess(reviews_train)

# test data processig
t_rev=preprocess(reviews_test)
        


#create CSR matrix by implementing inverse document frequency and l2 normalization

vectorizer = TfidfVectorizer(norm = 'l2',min_df = 0, use_idf = True, smooth_idf = False, sublinear_tf = True, \
                             ngram_range=(1,2), max_features = 9000 )

#vectorize train review
train_vect = vectorizer.fit_transform(review)
train_vect_show=train_vect[0:100,:]

train_vect = train_vect.toarray()
print (train_vect.shape)


test_vect = vectorizer.transform(t_rev)

test_vect = test_vect.toarray()
print (test_vect.shape)


k = 300
test_sentiments = list()

#get cosine similarity
similarities = cosine_similarity(test_vect,train_vect)
similarities_show=similarities[0:100,:]




for similarity in similarities:
    
    #Nearest neighbor find
    knn = findKN(similarity, k)
    prediction = reviewCh(knn, sentiments_train)
    
    #To write to the list as +1 instead of just a 1 for positive reviews
    if prediction == 1:
        test_sentiments.append('1')
    else:
        test_sentiments.append('-1')



with open('format.txt', 'w') as log:
      for x in test_sentiments:
          log.write(x+'\n')                
