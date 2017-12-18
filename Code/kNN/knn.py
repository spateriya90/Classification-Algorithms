
# coding: utf-8

# In[53]:

from collections import Counter
import math as math
import operator
import itertools
import numpy as np

def populate_data(filename):                        #Function to Read data from Input File and store it
    f = open(filename,'r')
    data = []
    for line in f:
        #data.append()
        l = []                                      #Create temp list to store line
        for columns in line.split():                #Iterate over all values 
            try:
                val = float(columns)                #To check if data is continous(float,int, etc) or categorical(string,char,etc)
                l.append(val)                       #Append as float
            except ValueError:
                l.append(columns)                   #Append as String or Char
        data.append(l)                             
    return data                                    #Return the list of lists containing data from the file

def remove_column(matrix, column):                 #Function to remove last column from Data before Normalizing it
    return [row[:column] + row[column+1:] for row in matrix]

def normalized_data(data):                         #Function which normalizes data
    col = len(data[0])                             #Get number of attributes ( columns )
    
    last_column = [row[-1] for row in data]        #Store last Column containing labels
    
    data = remove_column(data,col-1)               #Remove last column of the data
    
    new_data = []

    colList = []
    
    for c in range(0,len(data[0])):            
        try:
            clist = [ float(data[i][c]) for i in range(len(data)) ]         #Iterate over all columns of the data
            mean = np.mean(clist)
            std = np.std(clist)
            alist = []
            for x in clist:
                alist.append((x-mean)/std)                                  #Perform normalization
            colList.append(alist)
        except:
            colList.append([ data[i][c] for i in range(len(data)) ])        #Dont normalize if string
    colList = list(map(list, zip(*colList)))                                #Take transpose
    new_data = np.insert(colList, col-1, last_column, axis=1)               #Add last column of labels to normalized data
    newdata = []
    for line in new_data:                                                   #Convert normalized data to float and string as required
        l = []                                                              #Create temp list to store line
        for columns in line:                                                #Iterate over all values 
            try:
                val = float(columns)
                l.append(val)
            except ValueError:
                l.append(columns)
        newdata.append(l)
    return newdata                                                          #return new Normalized Data

def split_data(data,iteration,no_of_folds):                                 #Function to perform k-fold cross validation
    
    testing_data = []
    training_data = []
    batch = len(data)/no_of_folds                                           #Size of each batch(unit of data)
    
    for i,j in enumerate(data):                                             #Distribute data into test and training
        if(iteration==1):
            if(i>=0 and i<int(iteration*batch)):
                testing_data.append(j)
            else:
                training_data.append(j)
        elif(iteration==no_of_folds):
            if(i>=int((iteration-1)*batch) and i<=len(data)-1):
                testing_data.append(j)
            else:
                training_data.append(j)
        else:
            if(i<int(batch*(iteration-1)) or i>=int(iteration*batch)):
                training_data.append(j)
            else:
                testing_data.append(j)
    
    return training_data, testing_data                                     #Return training and test data
    
def calculate_distance(train_row,curr_row):                                #Function to calculate euclidean distance 
    dist = 0;
    no_of_attr = len(curr_row)-1
    for train_val,curr_val in zip(train_row,curr_row):                    #enumerate 
        if(no_of_attr>0):
            if(isinstance(curr_val, float)):                             #check if float i.e. continous data
                dist = dist + math.pow(float(train_val)-float(curr_val),2)
            else:                                                        #when data is categorical
                if(train_val!=curr_val):                                 #if categorical values are not equal
                    dist = dist + 1
        no_of_attr=no_of_attr-1                                          #loop over all attributes
    return math.sqrt(dist)                                               #return the distance

def findknn(training_data, curr_row, k):                                 #function to find k-nearest neighbors of given test data value
    distance = []
    row_number=1
    for train_row in training_data:
        dist = calculate_distance(train_row,curr_row)                   #calculate distance between test data value and training data value
        distance.append((train_row,dist))                               #create a distance list
        row_number=row_number+1
    distance.sort(key=lambda x: x[1])                                   #sort the distances 
    neighbors = []
    distance = distance[:k]                                             #select top k (i.e. k nearest neighbors)
    neighbors = [item[0] for item in distance]                          #get the neighbors based on distances
    return neighbors                                                    #return the list of nearest neighbors


def count_max_label(neighbors):                                         #function to predict label of the test data
    label_val = []
    for val in neighbors:                                               #extract label of all the nearest neighbors
        label_val.append(val[-1])
    most_common,num_most_common = Counter(label_val).most_common(1)[0]  #select the most common label
    return most_common                                                  #return the selected label for the current test data value
        
def statistics(test_labels, predicted_labels):                          #function to calculate the performance metrics

    pos = 0
    neg = 0
    false_pos = 0
    false_neg = 0
    acc=0
    precision =0
    recall =0
    f_measure=0
    for x,y in zip(test_labels,predicted_labels):                       #enumerate
        #print(x,"   ",y)
        if(x == y):                                                     #to populate TP,TN,FP,FN values
            if(x == 1):
                pos = pos + 1
            else:
                neg = neg + 1
        else:
            if(x == 1 and y == 0):
                false_neg = false_neg + 1
            elif(x == 0 and y == 1):
                false_pos = false_pos + 1
        if((pos+neg+false_pos+false_neg)!=0):                                #to calculate accuracy
            acc = (pos+neg) / float(pos+neg+false_pos+false_neg)
        else:
            acc = pos+neg
        if((pos+false_pos)!=0):                                              #to calculate precision
            precision = pos / float(pos + false_pos)
        else:
            precision = pos
        if((pos+false_neg)!=0):                                              #to calculate recall
            recall = pos / float(pos + false_neg)
        else:
            recall = pos
        if((precision + recall)!=0):                                         #to calculate f-measure
            f_measure = (2 * precision * recall) / float(precision + recall)
        else:
            f_measure = (2 * precision * recall)
    return acc, precision, recall, f_measure                                 #return the calculated values

print("Starting K-Nearest Neighbor Classification: ")
response = input('Do you want to input multiple (test or train) files for Demo Data (Y/y) or perform Cross Validation on input data set(N/n)? (y for  n): ')

if(response == 'n' or response == 'N'):
    filename = input('Enter file name which contains Data: ')                  #read input file name containing the data
    #print(filename)
    k = int(input('Enter number of Nearest Neighbors (k): '))                  #read value of k (number of nearest neighbors)
    #k=2
    d = populate_data(filename)                                                #populate the data
    data = normalized_data(d)                                                  #normalize the data
    no_of_folds = 10
    acc = 0
    accuracy = 0
    precision = 0
    recall = 0
    f_measure = 0

    for i in range(no_of_folds):                                              #run 10 times - no of folds
        iteration = i+1
        print("\nFold Number:  " + str(iteration))
        training_data, testing_data = split_data(data,iteration,no_of_folds)  #split the data into training and test data
        predicted_labels = []
        test_labels = []
        for curr_row in testing_data:                                         #iterate over test data
            neighbors = findknn(training_data, curr_row, k)    #find the nearest neighbors of current test data
            label = count_max_label(neighbors)                                #find the label 
            predicted_labels.append(label)                                    #add the found label to the list of predicted labels
        test_labels = [item[-1] for item in testing_data]
        acc, prec, rec, f = statistics(test_labels, predicted_labels)         #calculate performance metrics depending on test labels and predicted labels
        accuracy = accuracy + acc
        precision = precision + prec
        recall = recall + rec
        f_measure = f_measure + f
        print("\nPerformance Metrics for Iteration : " + str(i+1))
        print("Accuracy : " + str(round(acc, 4)*100)+"%" + "\tPrecision : " + str(round(prec,4)*100)+"%" + "\tRecall : " + str(round(rec,3)*100)+"%" + "\t\tF-1 Measure : " + str(round(f,4))) 
    accuracy = accuracy / float (no_of_folds)
    precision = precision / float (no_of_folds)
    recall = recall / float (no_of_folds)
    f_measure = f_measure / float (no_of_folds)
    print("\nPerformance Metrics Obtained After " + str(no_of_folds)+ "-Fold Cross Validation for k = "+str(k))
    print("Average Accuracy : " + str(round(accuracy, 4)*100)+"%" + "\tAverage Precision : " + str(round(precision,4)*100)+"%" + "\tAverage Recall : " + str(round(recall,4)*100)+"%" + "\t\tAverage F-1 Measure : " + str(round(f_measure,3))) 
else:
    training_data = populate_data(input('Enter file name which contains Training Data: '))            #get training data from user
    testing_data = populate_data(input('Enter file name which contains Testing Data: '))              #get testing data from user
    k = int(input('Enter number of Nearest Neighbors (k):'))                                          #get value of k
    accuracy = 0
    precision = 0
    recall = 0
    f_measure = 0
    predicted_labels = []
    test_labels = []
    for curr_row in testing_data:                                                                     #iterate over testing data values
        neighbors = findknn(training_data, curr_row, k)                                               #find neighbors
        label = count_max_label(neighbors)                                                            #find label of test data value
        predicted_labels.append(label)                                                                #add to list of predicted labels
    test_labels = [item[-1] for item in testing_data]
    acc, prec, rec, f = statistics(test_labels, predicted_labels)                                     #calculate performance metrics based on test labels and predicted labels 
    print("\nPerformance Metrics for k = "+str(k)+" are :")
    print("Accuracy : " + str(round(acc, 4)*100)+"%" + "\tPrecision : " + str(round(prec,4)*100)+"%" + "\tRecall : " + str(round(rec,4)*100)+"%" + "\t\tF-1 Measure : " + str(round(f,3))) 


# In[ ]:



