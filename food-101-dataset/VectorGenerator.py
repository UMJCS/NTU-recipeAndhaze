import os
import numpy as np
def Vector():
    orderfile = open('vector_order.txt','a')
    vectorfile = open('foodtype_vectors.txt','a')
    with open( './ingredients-101/annotations/ingredients_simplified.txt','r') as f:
        data = f.readlines()
        for line in data:
            word = line[:-1].split(',')
            for i in word:
                if i not in words_vector:
                    words_vector.append(i)
                    orderfile.write(i+'\n')
        for line in data:
            temp_vector = np.zeros(len(words_vector))
            word = line[:-1].split(',')
            for i in word:
                index = words_vector.index(i)
                temp_vector[index]+=1
            foodtype_vector.append(temp_vector)
            vectorfile.write(str(temp_vector)+'\n')
    f.close()
    orderfile.close()
    vectorfile.close()







def Generator(type):
    with open(type +'_label.txt','r') as f:
        data = f.readlines()
        s= data[0]



RootPath = os.getcwd()
words_vector = []
foodtype_vector = []
Vector()
