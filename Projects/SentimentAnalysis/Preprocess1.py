import os
import pickle
import numpy as np
import random
#0-pos
#1-neg
training_data = []
word_of_bag = set()
numwords = []
def create_training_data():
    DIRECTORY = "C:\\Users\\v.vatsalya\\.spyder-py3\\aclImdb\\train"
    CATEGORIES = ["pos", "neg"]
    
    for categories in CATEGORIES:
        path = os.path.join(DIRECTORY, categories)
        class_label = CATEGORIES.index(categories)
        for i in os.listdir(path):
            try:
                training_data.append([os.path.join(path,i),class_label])
            except Exception as e:
                pass
            
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding = 'utf-8') as f:
        words = []
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.append(curr_word)
            word_to_vec_map[(curr_word)] = np.array(line[1:], dtype = np.float64)          
    return [words, word_to_vec_map]

lis = read_glove_vecs('glove.6B/glove.6B.50d.txt')
wordVectors1 = (lis[1])
wordVectors = np.zeros((400000,50))
i = 0
for val in wordVectors1.values():
    wordVectors[i] = val
    i = i + 1
np.save("gloveVectors.npy", wordVectors)
np.save("glove.npy", lis[0])


create_training_data()
print(sum(numwords)/len(numwords))
print(len(numwords))
X=[]
y=[]       
random.shuffle(training_data)
for feature,label in training_data:
    X.append(feature)
    y.append(label)

pickle_out = open('X_train.pickle','wb')
pickle.dump(X, pickle_out)
pickle_out = open('y_train.pickle','wb')
pickle.dump(y, pickle_out)


