import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import math

# Guide followed from https://towardsdatascience.com/wondering-how-to-build-an-anomaly-detection-model-87d28e50309


# load our dataset
dataset = sio.loadmat('anomalyData.mat')
X = dataset['X']
Xval = dataset['Xval']
yval = dataset['yval']

# dataset visualization
#plt.scatter(X[:, 0], X[:, 1], marker = "x")
#plt.xlabel('Latency(ms)')
#plt.ylabel('Throughput(mb/s)')

def estimateGaussian(X):
    n = np.size(X, 1)
    m = np.size(X, 0)
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))
    
    mu = np.reshape((1/m)*np.sum(X, 0), (1, n))
    sigma2 = np.reshape((1/m)*np.sum(np.power((X - mu),2), 0),(1, n))
    
    return mu, sigma2


mu, sigma2 = estimateGaussian(X)
print('mean: ',mu,' variance: ',sigma2)

def multivariateGaussian(X, mu, sigma2):
     n = np.size(sigma2, 1)
     m = np.size(sigma2, 0)
     #print(m,n)
     
     if n == 1 or m == 1:        
         sigma2 = np.diag(sigma2[0, :])
     
     X = X - mu
     pi = math.pi
     det = np.linalg.det(sigma2)
     inv = np.linalg.inv(sigma2)
     val = np.reshape((-0.5)*np.sum(np.multiply((X@inv),X), 1),(np.size(X, 0), 1))
     
     p = np.power(2*pi, -n/2)*np.power(det, -0.5)*np.exp(val)
     
     return p


p = multivariateGaussian(X, mu, sigma2)
print(p.shape)
pval = multivariateGaussian(Xval, mu, sigma2)

def selectThreshHold(yval, pval):
    
    F1 = 0
    bestF1 = 0
    bestEpsilon = 0
    
    stepsize = (np.max(pval) - np.min(pval))/1000
        
    epsVec = np.arange(np.min(pval), np.max(pval), stepsize)
    noe = len(epsVec)
    
    for eps in range(noe):
        epsilon = epsVec[eps]
        pred = (pval < epsilon)
        prec, rec = 0,0
        tp,fp,fn = 0,0,0
        
        try:
            for i in range(np.size(pval,0)):
                if pred[i] == 1 and yval[i] == 1:
                    tp+=1
                elif pred[i] == 1 and yval[i] == 0:
                    fp+=1
                elif pred[i] == 0 and yval[i] == 1:
                    fn+=1
            prec = tp/(tp + fp)
            rec = tp/(tp + fn)
            F1 = 2*prec*rec/(prec + rec)
            if F1 > bestF1:
                bestF1 = F1
                bestEpsilon = epsilon
        except ZeroDivisionError:
            print('Warning dividing by zero!!')          
       
    return bestF1, bestEpsilon



F1, epsilon = selectThreshHold(yval, pval)
print('Epsilon and F1 are:',epsilon, F1)
outl = (p < epsilon)

def findIndices(binVec):
    l = []
    for i in range(len(binVec)):
        if binVec[i] == 1:
            l.append(i)
    return l


listOfOutliers = findIndices(outl)
count_outliers = len(listOfOutliers)
print('\n\nNumber of outliers:', count_outliers)
print('\n',listOfOutliers)

#plt.scatter(X[:, 0], X[:, 1], marker = "x")
#plt.xlabel('Latency(ms)')
#plt.ylabel('Throughput(mb/s)')
#plt.scatter(X[listOfOutliers,0], X[listOfOutliers, 1], facecolors = 'none', edgecolors = 'r')
#plt.show()


# Using a larger dataset
newDataset = sio.loadmat('anomalyDataTest.mat')

Xtest = newDataset['X']
Xvaltest = newDataset['Xval']
yvaltest = newDataset['yval']

plt.scatter(Xtest[:, 0], Xtest[:, 1], marker = "x")
plt.xlabel('Latency(ms)')
plt.ylabel('Throughput(mb/s)')

print(Xtest.shape)
print(Xvaltest.shape)
print(yvaltest.shape)

mutest, sigma2test = estimateGaussian(Xtest)

ptest = multivariateGaussian(Xtest, mutest, sigma2test)

pvaltest = multivariateGaussian(Xvaltest, mutest, sigma2test)


F1test, epsilontest = selectThreshHold(yvaltest, pvaltest)
print('\nBest epsilon and F1 are\n',epsilontest, F1test)

outliersTest = ptest < epsilontest
listOfOl = findIndices(outliersTest)

plt.scatter(Xtest[:, 0], Xtest[:, 1], marker = "x")
plt.xlabel('Latency(ms)')
plt.ylabel('Throughput(mb/s)')
plt.scatter(Xtest[listOfOl,0], Xtest[listOfOl, 1], facecolors = 'none', edgecolors = 'r')
plt.show()



print('\n\n Outliers are:\n',listOfOl)
print('\n\nNumber of outliers are: ',len(listOfOl))










