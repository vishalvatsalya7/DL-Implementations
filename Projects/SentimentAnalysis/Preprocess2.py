import time
import pickle
import numpy as np
import random
import tensorflow as tf
import re
from os import listdir
from os.path import isfile, join

X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cosine_similarity(u, v):
    dot = np.dot(u,v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cosine_similarity = dot/(norm_u*norm_v)
    return cosine_similarity

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

wordsList = np.load('glove.npy').tolist()
wordVectors = np.load('gloveVectors.npy')

maxSeqLength = 234 #Maximum length of sentence
numfile = 25000

ids = np.zeros((numfile, maxSeqLength), dtype = 'int32')
fileCounter = 0
for pf in X_train:
    with open(pf, "r", encoding='utf-8') as f:
        indexCounter = 0
        line=f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 399999 
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        fileCounter = fileCounter + 1 
np.save('idsMatrix.npy', ids)

    


