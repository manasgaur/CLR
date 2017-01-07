##### implementation of convolutional logistic regression ############

###### Accuracy of Logistic Regression is better than Neural Net on Small Data Set ---- Inquire ###########
##### this is a simple implementation without normalization and outlier removal ###########

from sklearn.decomposition import TruncatedSVD
import numpy as Math
import csv
from sklearn import preprocessing
from matplotlib import pyplot as plt
from random import randrange
from random import seed
import random

# in this function I will be sending the matrix "new_mat" and will
def CalculateConvolution(Matrix):
    boxsize=2
    Convmat= [[0 for x in range(len(Matrix[0]))] for y in range(len(Matrix))]
    print Math.asarray(Convmat).shape
    #print Matrix[226][9]
    n=9
    for i in range(len(Matrix)-1):
        if i == 0:
            for j in range(len(Matrix[0])-1):
                if j!=len(Matrix[0])-1:
                    for k in range(1,boxsize):
                        Convmat[i][j]=Matrix[i][j]*(Matrix[i+k][j]+ Matrix[i][j+k] + Matrix[i+k][j+k])
                else:
                    for k in range(1,boxsize):
                        Convmat[i][j]=Matrix[i][j]*(Matrix[i-k][j]+ Matrix[i][j-k] + Matrix[i-k][j-k])
        elif i == len(Matrix):
            for j in range(len(Matrix[0])-1):
                if j!=len(Matrix[0])-1:
                    for k in range(1,boxsize):
                        Convmat[i][j]=Matrix[i][j]*(Matrix[i+k][j]+ Matrix[i][j+k] + Matrix[i-k][j-k])
                else:
                    for k in range(1,boxsize):
                        Convmat[i][j]=Matrix[i][j]*(Matrix[i-k][j]+ Matrix[i][j-k] + Matrix[i-k][j-k])
        else:
            for j in range(len(Matrix[0])-1):
                for k in range(1,boxsize):
                    Convmat[i][j] = Matrix[i][j]*(Matrix[i][j+k]+Matrix[i][j-k]+Matrix[i+k][j]+Matrix[i-k][j]+Matrix[i-k][j+k]+ Matrix[i+k][j-k]+Matrix[i-k][j-k]+Matrix[i+k][j+k])
    return Convmat

#sample=Math.random.random_sample((10,))
#print Math.sum(Math.convolve(new_mat[0][:],new_mat[:][10],'same'))


def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + Math.exp(-yhat))
'''
def predict2(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return Math.exp(yhat)
'''

# Estimate logistic regression coefficients using stochastic gradient descent
error_list=[]
def coefficients_sgd(train,l_rate,n_epoch,type):
    #temp=[]
    coef=[0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error=0
        for row in train:
            if type==1:
                yhat=predict(row,coef)
            #else:
                #yhat=predict2(row,coef)#
            error=row[-1] - yhat
            sum_error-=error**2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0-yhat)
            for i in range(len(row)-1):
                coef[i+1]= coef[i+1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    error_list.append(sum_error)
    print ('epoch=%d, lrate=%.3f, error=%.3f' %(epoch,l_rate,sum_error))
       # temp.append(sum_error)
    return coef

def CLR(new_mat, test, l_rate, epochs,ls):
    predictions=[]######################## LAYER 1########################################
    HN1=ls
    layer1=[]
    for i in range(HN1):
        layer1.append(coefficients_sgd(CalculateConvolution(new_mat),l_rate,epochs,1))
        ''' new_mat=Math.vstack([CalculateConvolution(new_mat),result])'''
    layer2=[]
    HN2=1
    '''for j in range(HN2):'''
    ###### If this layer is not convoluted then there is accuracy drops
    layer2=(coefficients_sgd(layer1,l_rate,epochs, 1))
    '''new_mat=Math.vstack([CalculateConvolution(new_mat),result2])'''
###########################################################################################
    coef=layer2
    #coef=coefficients_sgd(layer2,0.01,epochs,2)
    for row in test:
        yhat=predict(row,coef)
        yhat=round(yhat)
        predictions.append(yhat)
    return predictions

######## performing cross-validation##########################
######## alternative is using Sklearn #########################
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = len(dataset) / n_folds
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


########### we can test all the algorithms here. For sample we are using logistic regression####################

def eval_algorithm(dataset, algorithm, n_folds, l_rate,n_epoch,ls):
    folds=cross_validation_split(dataset,n_folds)
    scores=[]
    for f in folds:
        train=list(folds)
        train.remove(f)
        train=sum(train,[])
        test= list()
        for row in f:
            row_copy = list(row)
            test.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train, test, l_rate, n_epoch, ls)
        actual = [row[-1] for row in f]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
	return scores




# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
dataset=[]
with open('location of CSV file','r') as f:
    readr=csv.reader(f,delimiter=',')
    for row in readr:
        if not row:
            continue
        dataset.append(row)

for r in range(len(dataset)):
    for c in range(len(dataset[0])):
        dataset[r][c]= float(dataset[r][c].strip())


# normalize
scale=preprocessing.MinMaxScaler()
normalize_set=scale.fit_transform(Math.asarray(dataset))
# evaluate algorithm
n_folds = 2
l_rate = 0.001
n_epoch = 100
latent_states=100
scores = eval_algorithm(normalize_set, CLR, n_folds, l_rate, n_epoch, latent_states)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))+2))
#initial_input=plt.plot(range(epochs),error_list[0],'r-')
#layer1=plt.plot(range(epochs),error_list[1],'b*')
#layer2=plt.plot(range(epochs),error_list[2],'g--')
#plt.show()
print len(error_list)

plt.plot(range(latent_states+1), error_list, 'r-')
plt.xlabel('latent states learning')
plt.ylabel('error')
plt.show()



######### we are not including this following layer as this will increase the error ######
'''new_mat=Math.vstack([CalculateConvolution(new_mat),result3])
result4=coefficients_sgd(CalculateConvolution(new_mat),0.01,epochs)'''
###########################################################################################
#print Math.asarray(result).shape