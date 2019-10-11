import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
	Z=np.exp(X)
	Z=Z+1
	Z=np.reciprocal(Z)
	return Z

def accuracy(Y,y_pred):
	k=np.zeros((number_of_inputs,1))
	for i in range(number_of_inputs):
		if(y_pred[i,0] > 0.5):
			k[i]=1
	accuracy=np.sum(k*Y+(1-k)*(1-Y))/number_of_inputs
	return accuracy,k

#Data Handling
f = open("ex2data1.txt", "r")
s=""
s=f.readline();
data=np.zeros((100,3))
i=0
while(s):
	#print(s)
	s=f.readline()
	tt=[]
	if(s):
		tt=s.split(',')
		for j in range(3):
			data[i,j]=float(tt[j])
		i+=1

X=data[:,0:2]
Y=data[:,2:]
number_of_inputs, number_of_features=data.shape

#Weight Initialization
W=np.random.rand(2,1)*0.001

#Hyperparameters
learning_rate=1e-1
max_iter=100

#Training
for i in range(max_iter):
	cost=0
	h=X.dot(W)
	y_pred=sigmoid(h)
	cost=-np.sum(Y*(np.log(y_pred+1e-5))-(1-Y)*(np.log(1-y_pred+1e-5)))
	cost/=number_of_inputs
	grad=h-Y
	grad=((np.transpose(X)).dot(grad))/number_of_inputs
	W=W-(learning_rate*grad)
	acc, k=accuracy(Y,y_pred)
	print(acc)

#Data
for i in range(number_of_inputs):
	if(Y[i]==1):
		plt.plot(X[:,0],X[:,1],'r+')
	else:
		plt.plot(X[:,0],X[:,1],'bo')	

#Pred
for i in range(number_of_inputs):
	if(k[i]==1):
		plt.plot(X[:,0],X[:,1],'r+')
	else:
		plt.plot(X[:,0],X[:,1],'bo')	
#plt.show()


