# Import required packages
import keras
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model
from matplotlib import pyplot

# import mnist dataset for keras
from keras.datasets import mnist

# define Global Constants
batch_size = 100

# load MNIST dataset
(x_train, Y_train), (x_test, Y_test) = mnist.load_data()

# plot the first 9 MNIST data
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
pyplot.suptitle("First 9 MNIST Digits")
pyplot.show()

# normalize the input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# convert the categorical output to one-hot-code
y_test = keras.utils.to_categorical(Y_test, 10)
y_train = keras.utils.to_categorical(Y_train, 10)

# Build the Sequential Model
model = Sequential()

# Add a Flatten Layer to flatten the 28x28 MNIST image
model.add(Flatten(input_shape=(28, 28, 1)))

# Add the hidden layers with a ReLU activation
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

# Add a dense output layer with 10 neurons (for numbers 0 to 9) with a softmax activation
model.add(Dense(10, activation='softmax'))

# Print the layers in the model
model.layers

# Print the model summary
model.summary()

# Compile the model using the adam optimizer
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the number of epochs
num_epochs = 50

# Fit the model
fit_history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


# Example predicting a digit
random_index = random.randint(0, Y_test.shape[0])
print("Current Number is: ", Y_test[random_index])
test_input = x_test[random_index]
pyplot.imshow(test_input, cmap=pyplot.get_cmap('gray'))
pyplot.suptitle("Test Digit")
pyplot.show()
output_prob = np.array(model.predict(test_input.reshape(1, 28, 28))[0])
output_pred = np.argmax(output_prob)
print("Predicted Number is: ", output_pred)

