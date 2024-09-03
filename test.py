import csv
import random
import cv2
import os
import sys
from network import *
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

NUM_CATEGORIES = 3
IMG_WIDTH = 2
IMG_HEIGHT = 2

def load_data(data_dir):
    
    images = []
    labels = []
    for i in range(NUM_CATEGORIES):
        for j in os.listdir(os.path.join(data_dir, str(i))):
            path = os.path.join(data_dir, str(i), j)
            #print('[DEBUG] path:', path)

            
            if os.path.isfile(path):
                img = cv2.imread(path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_NEAREST)
                images.append(img)
                labels.append(i)


    return (images, labels)


def test_image(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_NEAREST)
	return img

#(images, labels) = load_data("colors")
#print(images)
#print(labels)

#x_train = np.zeros((len(images), IMG_HEIGHT * IMG_WIDTH * 3))
#y_train = np.zeros((len(images), NUM_CATEGORIES))

#for i in range(len(images)):
#	x_train[i] = images[i].flatten()
#	y_train[i][labels[i]] = 1

#print(x_train)
#print(y_train)


data = load_iris()
print(data)
x, y = data["data"], data["target"]
x_train,x_test,y_train_single,y_test = train_test_split(x, y, test_size = 0.2)
y_train = np.zeros((len(y_train_single), 3))
for i in range(len(y_train_single)):
	y_train[i][y_train_single[i]] = 1

y_train = y_train.T
model = Model()


model.add_input_layer(InputLayer(4))
model.add_layer(Layer(6, "relu"))
model.add_layer(Layer(6, "relu"))
model.add_layer(OutputLayer(3))
#model.add_layer(Layer(NUM_CATEGORIES, "softmax"))
#model.print()


model.train(x_train, y_train, epochs=10)
print("DSKAFOSOHJJF")
print(x_test)
#y_pred = model.predict(np.array([[5, 3.5, 1.6, 0.6]])) #0, 1
#y_pred_two = model.predict(np.array([[4.9, 2.4, 3.3, 1]]))
y_pred = model.predict(x_test)
print(y_pred.T)
#print(y_pred_two.T)
print(y_test)
print(y_train.T)

#x_train = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6], [11, 12, 13], [15, 16, 17], [12, 13, 14], [10, 11, 12]])
#y_train = np.array([2, 4 ,5, 12, 16, 13, 11])

#x_train = np.zeros((size, 3))
#y_train = np.zeros(size)
"""
for i in range(size):
	x = random.randint(0, 1000)
	x_train[i] = np.array([x - 1, x, x + 1])
	y_train[i] = x
print(x_train)
print(y_train)
"""

#model.train(x_train, y_train, epochs=10)

#img = test_image("test/646464.png").flatten().reshape(1, IMG_HEIGHT*IMG_WIDTH*3)
#print(img)
#print(len(img))
#model.predict(img)




"""
model = Model()
model.add_input_layer(InputLayer(4))
model.add_layer(Layer(3, "relu"))
model.add_layer(OutputLayer(1))

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# Separate data into training and testing groups
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# Train model on training set
X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
print(X_training)
print(y_training)
model.train(X_training, y_training)
"""
