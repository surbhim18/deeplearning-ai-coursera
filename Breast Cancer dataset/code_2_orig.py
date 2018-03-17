﻿import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("classes: " + str(classes))
print ('\n' + "-------------------------------------------------------" + '\n')


max_val_train = np.max(train_set_x_orig,axis =0) # (1,30)
max_val_test = np.max(test_set_x_orig,axis =0) # (1,30)

train_set_x_norm = train_set_x_orig/max_val_train
test_set_x_norm = test_set_x_orig/max_val_test

train_set_x = train_set_x_norm.T
test_set_x = test_set_x_norm.T

print(train_set_x)

def sigmoid(z):

	s = 1.0/(1.0+np.exp(-z))

	return s

	
def propagate(w, b, X, Y):

	m = X.shape[1]

	A = sigmoid(np.dot(w.T, X) + b)
	cost = (-1.0/m)*np.sum((Y*np.log(A)+ (1-Y)*np.log(1-A)), axis = 1)
	dw = (1.0/m)*np.dot(X, (A-Y).T)
	db = (1.0/m)*np.sum(A-Y, axis = 1)
	
	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	
	grads = {"dw": dw, "db": db}    
	
	return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

	costs = []

	for i in range(num_iterations):

		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		
		w = w - learning_rate*dw
		b = b - learning_rate*db
		
		if i % 100 == 0:
			costs.append(cost)
			
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))

	params = {"w": w,
			  "b": b}

	grads = {"dw": dw,
			 "db": db}

	return params, grads, costs

def predict(w, b, X):

	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X) + b)

	p = np.zeros(m).reshape(1,m)
	for i in range(A.shape[1]):

		if A[0,i]>0.5:
			Y_prediction[0,i] = 1


	assert(Y_prediction.shape == (1, m))

	return Y_prediction
	
	
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

	w, b = np.zeros(X_train.shape[0]).reshape(X_train.shape[0],1), 0.0

	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

	w = parameters["w"]
	b = parameters["b"]

	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


	d = {"costs": costs,
		 "Y_prediction_test": Y_prediction_test,
		 "Y_prediction_train" : Y_prediction_train,
		 "w" : w,
		 "b" : b,
		 "learning_rate" : learning_rate,
		 "num_iterations": num_iterations}

	return d

#end of functions	
	
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

print('\n' + "-------------------------------------------------------" + '\n')

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
	print ("learning rate is: " + str(i))
	models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = True)
	print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
	plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
