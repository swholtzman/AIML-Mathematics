import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist

# load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Training set: 60,000 samples; each sample is a 28x28 grayscale image
# Test set:     10,000 samples; each sample is a 28x28 grayscale image
print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)


# --------- PERSONAL NOTE 1 ---------

# FLATTENING
# 1. With each image as a 28x28 pixel image, we will end up with a vector with length of 784 per image
# 2. The new shape for the training will result in TWO 2D arrays (rectangular; 748 wide, 10,000-60,000 tall):
#       # X_train
#       - a 60,000 item array where each item is a 28x28 grid
#       - 60,000 rows with 784 columns
#       # X_test
#       - a 10,000 item array
#       - 10,000 rows (samples), 784 columns (features)
#
# ----- END OF  PERSONAL NOTE 1 -----


# flatten the images in X_train and X_test such that each sample becomes a row vector
# -------- ENTER YOUR CODE HERE --------
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
# -------- END YOUR CODE HERE --------

print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)



# --------- PERSONAL NOTE 2 ---------

# NORMALIZING
# 1. We must normalize all pixels to be 0 - 1
#   - Current values range from 0 to 255. Divide all values by 255 to get between 0 and 1
#
# ----- END OF  PERSONAL NOTE 2 -----

# convert the pixel values from integer to float32 and 
# normalize the pixel values from the range of 0-255 to 0-1
# -------- ENTER YOUR CODE HERE --------
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
# -------- END YOUR CODE HERE --------

print('X_train range: ', np.min(X_train), ', ', np.max(X_train))
print('X_test range: ', np.min(X_test), ', ', np.max(X_test))

# determine number of pixels in an input image
num_pixels = X_train.shape[1]

# determine number of classes
num_classes = 10


# --------- PERSONAL NOTE 3 ---------

# LAYERING
# 1. We must now define the layers of our neural network
#       - Each image is a flattened array of 748 rows (features). Input == 784
#       - ReLu is the standard "turn-on" function for hidden layers; Helps learn complex patterns
#       - SoftMax is used on the output layer for classification
#           - Takes the 10 raw scores from the final layer and converts to 10 probabilities adding to 1
#           - e.g., [0.01, 0.0, 0.02, 0.95, 0.0, 0.01, 0.0, 0.0, 0.01, 0.0]
#               - if 0-indexed, this would mean a 95% chance the image is the digit '3'
# ----- END OF  PERSONAL NOTE 3 -----


# define a deep neural network model using the sequential model API
# Layer 0: input layer specifying the dimension of each sample
# Layer 1: n^[1] = 800 nodes, g^[1] = ReLU
# Layer 2: n^[2] = 100 nodes, g^[2] = ReLU
# Layer 3: n^[3] = num_classes nodes, g^[3] = softmax
# -------- ENTER YOUR CODE HERE --------
model = Sequential()
model.add(Input((num_pixels, )))
model.add(Dense(800, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# -------- END YOUR CODE HERE --------

# print a summary of the model
model.summary()


# --------- PERSONAL NOTE 4 ---------

# COMPILING
# 1. Before training, we must use the listed Keras items and pass them to model.compile()
#       - SDG Optimizer: Engine for Model Improvement
#           - Find the bottom-- lowest error (analogy: hiker lost on mountain; find the valley!)
#               - Take a guess on a batch of images
#               - Loss function calculates how wrong the guess was (how far from the bottom)
#               - SDG looks at "slope" (gradient) of the "mountain at the hiker's position"
#               - Tell the "hiker" toward the steepest "downhill" direction
#           - Learning rate: size of step the "hiker" takes
#               - e.g., 0.0001 == tiny steps (high compute power needed); 2.0 == giant leaps (may overshoot the bottom)

#       - Sparse_Categorical_CrossEntropy:
#           - Error calculator ("scorecard"). Optimizer tries to make this as small as possible
#               - 1.Model's final SoftMax layer outputs a list of 10 probabilities ([0.05, 0.05, 0.1, 0.7, 0.05, ..., 0.05])
#               - 2.True answer from y_train is a single number (e.g., 3)
#               - 3.This loss function looks at the predicted probability for the correct answer
#                       - e.g., looks at index 3 and sees probability 0.7
#               - 4.Calculates "penalty" (loss). Closer the probability is to 1.0, the lower the penalty.
#                   - Closer to 0===higher penalty. Optimizer's job to update the model to make the probability closer to 1.0
#           - "Sparse"
#               - categorical_crossentropy: expects true answer to be a full "one-hot" vector
#                   - e.g., [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#               - sparse_categorical_crossentropy: used when answer is a single integer (e.g., 3)
#                   - saves us a step; y_train already looks like this

#       - Accuracy Metric:
#           - The human-readable report card
#           - Difference from "loss"
#               - Loss: complex continuous number (e.g., 0.2345). Great for SGD optimizer to use for calculating improvements
#                       - Not intuitive for us to read
#               - Accuracy: simple percentage. "What percentage of the time did the model's highest predicted probability match the correct answer?"
#           - Short: Train the model to minimize loss, evaluate the model's performance via the accuracy
# ----- END OF  PERSONAL NOTE 4 -----


# compile the model using
# a. Optimizer: gradient descent SGD with a learning rate of 0.1
# b. Loss function: sparse_categorical_crossentropy 
# c. Metrics: accuracy
# -------- ENTER YOUR CODE HERE --------
model.compile(optimizer=SGD(learning_rate=0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# -------- END YOUR CODE HERE --------
 
# fit the model to training data
model.fit(X_train, y_train, epochs = 10, verbose = 1)
 
# evaluate the model on the test data
loss, acc = model.evaluate(X_test, y_test, verbose = 1)
print('Test accuracy = %.3f' % acc)
