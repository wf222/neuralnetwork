import numpy as np
import math
import scipy
import mpmath
import csv
import matplotlib.pyplot as plt

class Layer():

	def __init__(self, nodes, activation):
		self.size = nodes
		self.values = np.random.randn(nodes, 1)
		self.activation = activation

	#Activations
	def relu(values):
		return np.maximum(0, values)

	def binary_step(values):
		
		return np.where(values >= 0, 1, 0)

	def sigmoid(values):
		temp = np.copy(values).astype(float)
		for i in range(len(temp)):
			for j in range(len(temp[i])):
				temp[i][j] = 1/(1 + math.e**(0-temp[i][j]))
		return temp

	def tanh(values):
		temp = np.copy(values).astype(float)
		for i in range(len(temp)):
			for j in range(len(temp[i])):
				temp[i][j] = (math.e**(temp[i][j]) - math.e**(0-temp[i][j]))/(math.e**(temp[i][j]) + math.e**(0-temp[i][j]))
		return temp

	"""
	def softmax(values):
		total = 0
		temp = np.copy(values).astype(float)
		for i in range(len(temp)):
			for j in range(len(temp[i])):
				print(temp[i][j])
				total += math.exp(temp[i][j])
		for i in range(len(temp)):
			for j in range(len(temp[i])):
				temp[i][j] = math.exp(temp[i][j])/total
				
		return temp
	"""

	def softmax(values):
		valuest = values.T
		temp = np.zeros_like(valuest)
		for i in range(len(valuest)):
			temp[i] = scipy.special.softmax(valuest[i])
		return temp.T

	#Derivatives
	def relu_deriv(values):
		return np.where(values >= 0, 1, 0)

	def binary_step_deriv(values):
		return np.zeros_like(values)

	def tanh_deriv(values):
		return 1 - Layer.tanh(values)**2

	def sigmoid_deriv(values):
		return Layer.sigmoid(values) * (1 - Layer.sigmoid(values))



	def activate(self):
		if self.activation == "relu":
			self.values = Layer.relu(self.values)
		elif self.activation == "binary_step":
			self.values = Layer.binary_step(self.values)
		elif self.activation == "sigmoid":
			self.values = Layer.sigmoid(self.values)
		elif self.activation == "tanh":
			self.values = Layer.tanh(self.values)
		elif self.activation == "softmax":
			self.values = Layer.softmax(self.values)


	def deriv(values, func):
		if func == "relu":
			return Layer.relu_deriv(values)
		elif func == "binary_step":
			return Layer.binary_step_deriv(values)
		elif func == "tanh":
			return Layer.tanh_deriv(values)
		elif func == "sigmoid":
			return Layer.sigmoid_deriv(values)
		elif func == "none":
			return values



class InputLayer(Layer):

	def __init__(self, nodes):
		super().__init__(nodes, None)

class ConvLayer(Layer):
	def __init__(self, input_size, filters=4):
		self.activation = None
		self.size = 2
		x, y = input_size
		self.values = np.zeros(10)


class OutputLayer(Layer):

	def __init__(self, nodes):
		super().__init__(nodes, "softmax")

class Model():

	def __init__(self):
		self.layers = []
		self.weights = []
		self.biases = []

	def add_input_layer(self, layer):
		self.layers.append(layer)
		self.weights.append(0)
		self.biases.append(0)

	def set_input(self, x_train):

		

		if len(self.layers) == 0:
			print("No input layer!")
			return

		self.layers[0].values = x_train.T

	def max_pool(values):
		temp = np.zeros_like(values)

		for i in range(len(values) - 1):
			for j in range(len(values[0]) - 1):
				temp[i][j] = max(values[i][j], values[i][j+1], values[i+1][j], values[i+1][j+1])

		return temp


	def add_layer(self, layer):

		if len(self.layers) == 0:
			print("No input layer!")
			return

		self.layers.append(layer)


		if isinstance(layer, ConvLayer):
			print("THIS CODE RANNNNNNNN")
			self.weights.append(0)
			self.biases.append(0)
		else:
			prev_length = self.layers[len(self.layers) - 2].size
			new_length = layer.size

			
			
			#self.weights.append(np.random.randn(new_length, prev_length) * 0.01)
			self.weights.append(np.full((new_length, prev_length), 0.001))
			self.biases.append(np.random.randn(new_length, 1))

	def print(self):
		for i in range(len(self.layers)):
			print()
			print("_____________")
			print("LAYER ", i)
			
			print("Nodes: ", self.layers[i].size)

	def propogate(self, layer_num):
		if isinstance(self.layers[layer_num], ConvLayer):
			#APPLY FILTERS
			print("POUQWOUQEWUPPUQRWOPURWQOPURQOPUW")
			return
		else:
			self.layers[layer_num].values = np.dot(self.weights[layer_num], self.layers[layer_num - 1].values) + self.biases[layer_num]
			self.layers[layer_num].activate()

	def forward_propagate(self):
		for i in range(1, len(self.layers)):
			#self.layers[i].values = np.dot(self.weights[i], self.layers[i - 1].values) + self.biases[i]
			#self.layers[i].activate()
			#print("LAYER " + str(i) + ": " + str(self.layers[i].values))
			self.propogate(i)


	def calc_grads(self, y_train):
		grad_weights = [0]
		grad_biases = [0]

		dA = 0
		dZ = 0

		for layer_num in range(len(self.layers) - 1, 0, -1):

			if isinstance(self.layers[layer_num], ConvLayer):
				grad_weights.insert(1, 0)
				grad_biases.insert(1, 0)
				continue

			m = len(y_train)
			if layer_num == len(self.layers) - 1:
				#Last Layer
				dA = 1/m * (self.layers[layer_num].values - y_train) #column vector
				dZ = dA #column vector
			else:
				#print("layer_num: ", layer_num)
				#print(dZ)
				#print(self.weights[layer_num + 1])
				dA = np.dot(self.weights[layer_num + 1].T, dZ) #column vector, same len as A
				dZ = np.multiply(dA, Layer.deriv(self.layers[layer_num].values, self.layers[layer_num].activation))


			
			grad_weights.insert(1, 1/m * np.dot(dZ, self.layers[layer_num - 1].values.T))
			grad_biases.insert(1, 1/m * np.sum(dZ))

		print("GRADIENTS")
		print(grad_weights)
		print()
		print("BIASES")
		print(grad_biases)

		return (grad_weights, grad_biases)

	def update(self, grad_weights, grad_biases, learning_rate):
		for i in range(1, len(self.layers)):
			self.weights[i] = self.weights[i] - learning_rate * grad_weights[i]
			self.biases[i] = self.biases[i] - learning_rate * grad_biases[i]

	def train(self, x_train, y_train, epochs=1000, learning_rate=0.01):
		self.set_input(x_train)
		
		y_train_transpose = np.transpose(np.atleast_2d(y_train))
		for i in range(epochs):
			print("Epoch ", (i + 1))
			self.forward_propagate()
			cost = Model.calc_cost(self.layers[len(self.layers) - 1].values, y_train)
			print("Cost at epoch " + str(i + 1) + ": " + str(cost) + "\n")


			(grad_weights, grad_biases) = self.calc_grads(y_train)
			self.update(grad_weights, grad_biases, learning_rate)

	def predict(self, x_predict):
		self.set_input(x_predict) # 4x2
		print(self.layers[0].values)
		self.forward_propagate()
		#self.print()
		return self.layers[len(self.layers) - 1].values

	def transpose_y(y_train):
		temp = np.zeros()

	def calc_cost(y_calc, y_train):
		cost = 1/(2*len(y_train)) * np.sum(np.square(y_calc - y_train))

		return cost





