import numpy as np
import pandas as pd
import math
import mldata
from data import *
from statistics import *

class LogisticRegression():
	def __init__(self,features_num, features_info, learning_rate = 0.1, weight_penalty = 0.1):
		self.features_info = features_info
		self.weights = np.random.random((features_num,1))
		self.bias = 0
		self.weight_penaty = weight_penalty
		self.learning_rate = learning_rate
	def classify(self, features):
		inner = np.matmul(features, self.weights) + self.bias
		p = sigmoid(inner)
		return p

	def classify_10(self, features):
		inner = np.matmul(features, self.weights) + self.bias
		p = sigmoid(inner)
		onezero = p > .5
		#onezero = onezero.astype(np.int)
		return onezero

	def loss(self,features, labels):
		#dimensions
		#features: items * features
		#labels: items * 1
		#self.weight: features * 1
		item_count = labels.shape[0]
		prediction = self.classify(features) # items* 1
		error_1 = -labels*np.log(prediction+0.000001)
		error_2 = -(1-labels)*np.log(1 - prediction+0.0000001)
		copy_weights = self.weights.copy()
		square_weight_sum = np.sum(self.weights**2) # scalar
		loss = error_1 + error_2
		loss = np.sum(loss)/item_count
		loss += self.weight_penaty * square_weight_sum
		return loss#returns scalar

	def getGrads(self, features, labels):
		item_count = labels.shape[0]
		prediction = self.classify(features)
		gradient_predictions = np.matmul(features.T, np.subtract(prediction ,
										 labels)) * 1/features.shape[0]#featers * items x items 1
		gradient_weights_sum = np.multiply(self.weights, self.weight_penaty)#featres*1
		gradient_weights = gradient_predictions + gradient_weights_sum #features * 1
		return gradient_weights#returns feature * 1

	def step(self, gradients, lr):
		self.weights = self.weights - lr*gradients

	def train(self, features, labels, iterations = 50):
		for epoch in range(iterations):
			loss = self.loss(features,labels)
			#print("Epoch {} Current Loss: {:.5f}".format(epoch,loss))
			self.step(self.getGrads(features,labels), self.learning_rate)

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))


dataset = input("Dataset options: volcanoes, spam, voting: ")
crossvalidation = input("Cross validation? 0/1: ")
# m = input("input value for m: ")
# num_buckets = input("input value for number of buckets: ")

data = None
if dataset == "volcanoes":
	data = read_data('..\\data\\volcanoes')
if dataset == "spam":
	data = read_data('..\\data\\spam')
if dataset == "voting":
	data = read_data('..\\data\\voting')

x, y, schema = c45_to_xy(data)
y = np.array(y)
y = y == 'True'
y = y.astype(np.int)
x = transform_nominal_attributes(x, schema)
# print(x)
sets = foldedCV(x, y, folds=5)
accuracies = [0] * 5
precisions = [0] * 5
recalls = [0] * 5
area_under_curves = [0] * 5
best_tree = None
best_accuracy = -1
for i in range(5):
	train_set, test_set = sets[i]
	train_x = [None] * len(train_set)
	train_y = [None] * len(train_set)
	test_x = [None] * len(test_set)
	test_y = [None] * len(test_set)

	for j in range(len(train_set)):
		train_x[j], train_y[j] = train_set[j]

	for j in range(len(test_set)):
		test_x[j], test_y[j] = test_set[j]

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	train_y = train_y.reshape(-1,1)


	n = LogisticRegression(len(schema), None, .1, 1)
	n.train(train_x, train_y, 20)
	predicted_labels = n.classify_10(test_x)
	accuracies[i] = accuracy(test_y, predicted_labels)
	precisions[i] = precision(test_y, predicted_labels)
	recalls[i] = recall(test_y, predicted_labels)
	area_under_curves[i] = area_under_curve(test_y, predicted_labels)

print('Accuracy: {:.3f} {:.3f}'.format( np.mean(accuracies), np.std(accuracies)))
print('Precision:{:.3f} {:.3f}'.format( np.mean(precisions), np.std(precisions)))
print('Recall: {:.3f} {:.3f}'.format( np.mean(recalls), np.std(recalls)))
print('Area under ROC: {:.3f}'.format(np.mean(area_under_curves)))