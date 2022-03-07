
# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks

# Step 1: Import all required keras libraries

from keras.datasets import mnist # This is used to load mnist dataset later
from keras.utils import np_utils # This will be used to convert your test image to a categorical class (digit from 0 to 9)
# imports for array-handling and plotting
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# for testing on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network
#from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
#from keras.utils import np_utils
import tensorflow as tf


# Step 2: Load and return training and test datasets
def load_dataset():
    # 2a. Load dataset X_train, X_test, y_train, y_test via imported keras library
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

	# 2b. reshape for X train and test vars - Hint: X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')

    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')


	# 2c. normalize inputs from 0-255 to 0-1 - Hint: X_train = X_train / 255
    X_train /= 255
    X_test /= 255
	# 2d. Convert y_train and y_test to categorical classes - Hint: y_train = np_utils.to_categorical(y_train)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
	# 2e. return your X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

# Step 3: define your CNN model here in this function and then later use this function to create your model

def digit_recognition_cnn():
	# 3a. create your CNN model here with Conv + ReLU + Flatten + Dense layers
    cnn = tf.keras.models.Sequential()
    
    cnn.add(tf.keras.layers.Conv2D(filters=30, kernel_size=5, activation='relu', input_shape = [28,28,1]))
    
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    cnn.add(tf.keras.layers.Conv2D(filters=15, kernel_size=3, activation='relu'))
    
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    cnn.add(tf.keras.layers.Dropout(0.2))
    
    cnn.add(tf.keras.layers.Flatten())
    
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=50, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    #cnn.summary()
	# 3b. Compile your model with categorical_crossentropy (loss), adam optimizer and accuracy as a metric
    cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    #cnn.fit(X_train, Y_train, batch_size = 128, epochs=20, verbose=2, validation_data=(X_test, y_test))
	# 3c. return your model
    return cnn

# Step 4: Call digit_recognition_cnn() to build your model
X_train, X_test, y_train, y_test = load_dataset()
model = digit_recognition_cnn()
    
# Step 5: Train your model and see the result in Command window. Set epochs to a number between 10 - 20 and batch_size between 150 - 200
    # training the model and saving metrics in history
history = model.fit(X_train, y_train, batch_size = 175, epochs=15, verbose = 0, validation_data=(X_test, y_test))
history.history
# Step 6: Evaluate your model via your_model_name.evaluate() function and copy the result in your report
model.evaluate(X_test, y_test, verbose = 0)

# Step 7: Save your model via your_model_name.save('digitRecognizer.h5')
# saving the model
save_dir = "/results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
del model

# Code below to make a prediction for a new image.

# Step 8: load required keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
# Step 9: load and normalize new image
def load_new_image(path):
	# 9a. load new image
    newImage = load_img(path, grayscale=True, target_size=(28, 28))
	# 9b. Convert image to array
    newImage = img_to_array(newImage)
	# 9c. reshape into a single sample with 1 channel (similar to how you reshaped in load_dataset function)
    #print(newImage.shape)
    newImage = newImage.reshape((newImage.shape[0], 28, 1)).astype('float32')
	# 9d. normalize image data - Hint: newImage = newImage / 255
    newImage = newImage / 255
	# 9e. return newImage
    return newImage

# Step 10: load a new image and predict its class
def test_model_performance():
    # 10a. Call the above load image function
    
    img = load_new_image('sample_images/digit5.png')
    
	# 10b. load your CNN model (digitRecognizer.h5 file)

    model = load_model('results/keras_mnist.h5')
    print("Loaded model from disk")

	# 10c. predict the class - Hint: imageClass = your_model_name.predict_classes(img)
    img = np.expand_dims(img, axis = 0)
    imageClass = model.predict_classes(img)
	# 10d. Print prediction result
    print("Result: ", imageClass[0])

# Step 11: Test model performance here by calling the above test_model_performance function
prediction = test_model_performance()
