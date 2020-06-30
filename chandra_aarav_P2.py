#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:35:57 2020

@author: aarav
"""
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

"""---------------------------------------------------------------------------"""
# Reads original file, shuffles and divides into train and test.
def read_shuffle_create():

    file_name = 'DivorceAll.txt'
    f_h = open(file_name,'r')
    # Reading first line of the file
    x = f_h.readline().split()
    m = int(x[0])
    n = int(x[1])
    
    # List comprehension for randomising data
    data = [ (random.random(), line) for line in f_h ]
    # Sorting data based on random values generated above
    data.sort()
    f_h.close()
    
    # Dividing data into 8:2 as train:test data.
    m_train = int(m*0.8)
    m_test = m - m_train
    
    # Writing total row and column values in each of the 2 files
    t_f = open('chandra_aarav_Train.txt','w')
    t_f.write(f"{m_train}\t{n}\n")
    te_f = open('chandra_aarav_Test.txt','w')
    te_f.write(f"{m_test}\t{n}\n")    
    
    ###### Dividing the data into Train and Test sets ######
    for i in range(m):
        # Train Data
        if (i<m_train):
            t_f.write(data[i][1])
        # Test Data
        else:
            te_f.write(data[i][1])
    t_f.close()
    te_f.close()   
    
"""---------------------------------------------------------------------------"""
def Logistic_Hypothesis(x,wT):
    z = (np.dot(x,wT))
    return(1/(1+(np.exp(-z)))) 
"""---------------------------------------------------------------------------"""
# Cost Function :
def cost(hw,y):
    epsilon = 1e-5
    return(-(y*np.log(hw+epsilon))-(1-y)*np.log(1-hw+epsilon))
"""---------------------------------------------------------------------------"""        
# For getting J(W):
def JW(c,m):
    return ((np.sum(c))/(m))
"""---------------------------------------------------------------------------"""
# New Weights:
def get_weights(w,alpha,hw,y,x):
    return (w-(alpha*(np.dot(((hw-y).T),x)).T))
"""---------------------------------------------------------------------------"""  
# used to shuffle and segregate the data into two files.
# read_shuffle_create()

# Reads the train file and puts it into corresponding matrix.      
file_name = input("Enter the Train File name: \n")
# file_name = "chandra_aarav_Train.txt"
f_h = open(file_name,'r')
x = f_h.readline().split()
m_train = int(x[0])
n_train = int(x[1])
train_tot=[(f_h.readline().strip('\n').split('\t')) for i in range(m_train)]
f_h.close()
    
train_tot = np.array(train_tot, dtype=float)
    
# Storing the shape to extract feature columns and result column
train_total_shape = np.shape(train_tot)

# Extracting all columns except the last column as 'features'
train_X = train_tot[:,:train_total_shape[1]-1]
    
# Adding 1 in the X
x0 = np.ones((m_train,1))
train_X = np.hstack([x0,train_X])
train_X_shape = np.shape(train_X)
    
# Extracting only the last column as 'result'
train_Y = train_tot[:,train_total_shape[1]-1]
train_Y = np.reshape(train_Y,(np.shape(train_Y)[0],1))
train_Y_shape = np.shape(train_Y)

"""---------------------------------------------------------------------------"""
# Computing weights using gradient descent

w = np.ones((train_X_shape[1],1))
    
j=[]
il = [i for i in range(1000)]
hw = Logistic_Hypothesis(train_X, w)

for i in range(1000):
    # cost
    cost_1 = cost(hw,train_Y)
    # error
    jw = JW(cost_1,train_X_shape[0])
    j.append(jw)
    # new weights
    n_w = get_weights(w,0.001,hw,train_Y,train_X)
    hw = Logistic_Hypothesis(train_X, n_w)
    w = n_w
print(f"\nInitial J for training data = {j[0]}\nFinal J for training data = {j[-1]}\n")
# Plotting     
plt.figure(figsize=(25,15))
sns.scatterplot(il,j)
plt.title('Iterations vs J')
# Set x-axis label
plt.xlabel('No. of iterations')
# Set y-axis label
plt.ylabel('J(W)')
plt.show()     
print(f"\nThe weights computed using gradient descent are as follows:\n{w}\n")

"""---------------------------------------------------------------------------"""
# Reads the test file and puts it into corresponding matrix.    

file_name = input("Enter the Test File name: \n")
# file_name = "chandra_aarav_Test.txt"
f_h = open(file_name,'r')
x = f_h.readline().split()
m_test = int(x[0])
n_test = int(x[1])
test_tot=[(f_h.readline().strip('\n').split('\t')) for i in range(m_test)]
f_h.close()
    
test_tot = np.array(test_tot, dtype=float)
    
# Storing the shape to extract feature columns and result column
test_total_shape = np.shape(test_tot)
    
# Extracting all columns except the last column as 'features'
test_X = test_tot[:,:test_total_shape[1]-1]
test_X_shape = np.shape(test_X)
    
# Adding 1 in the X
x0 = np.ones((m_test,1))
test_X = np.hstack([x0,test_X])
test_X_shape = np.shape(test_X)
    
# Extracting only the last column as 'result'
test_Y = test_tot[:,test_total_shape[1]-1]
test_Y = np.reshape(test_Y,(np.shape(test_Y)[0],1))
test_Y_shape = np.shape(test_Y)

# Compute j for the latest weights
cost_test = cost(hw,train_Y)
jw_test = JW(cost_test,train_X_shape[0])
print(f"\nJ for the test data = {jw_test}")

# Using the latest weights, we put it in Hypo to get results.
test_predictions = Logistic_Hypothesis(test_X, w)

"""---------------------------------------------------------------------------"""
# Changing the predictions for a definitive class. 
test_predictions = np.round(test_predictions)

tp,tn,fp,fn=0,0,0,0

for i,j in zip(test_predictions,test_Y):
    if(int(i)==1 and int(j)==1):
        tp+=1
    elif(int(i)==0 and int(j)==0):
        tn+=1
    elif(int(i)==1 and int(j)==0):
        fp+=1
    else:
        fn+=1
print(f"\nTrue Positive = {tp}\nTrue Negative = {tn}\nFalse Positive = {fp}\nFalse Negative = {fn}")

accuracy,precision,recall,f1 = 0.,0.,0.,0.

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = (tp)/(tp+fp) 
recall = (tp)/(tp+fn)
f1 = 2/((1/precision)+(1/recall))

print(f"\nAccuracy = {accuracy}\nPrecision = {precision}\nRecall = {recall}\nF1 Score = {f1}")

print ("\t\t\t\tPredicted Class")
print ("\t\t\t -------------------------")
print("\t\t\t |\t |N \t |Y\t|")
print ("\t\t\t -------------------------")
print(f"\tActual Class\t |N\t |{tn}\t |{fp}\t|")
print ("\t\t\t -------------------------")
print(f"\t\t\t |Y\t |{fn}\t |{tp}\t|")
print ("\t\t\t -------------------------")










