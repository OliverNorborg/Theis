print("******************************")
print("**** Beginning og program ****")
print("******************************")


# Resnet 32 for solar panel cells including image augmenation
print("Resnet 32 for solar panel cells including image augmenation")
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
import glob
import os
from PIL import Image, ImageOps
from PIL.ImageOps import flip, mirror

import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, AveragePooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras.backend as K

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

import joblib

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf
import numpy as np

models = tf.keras.models
layers = tf.keras.layers
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers


print("******************************")
print("*** All packegs are loaded ***")
print("******************************")

# Is only necessary when running in colab
#from google.colab import drive
#drive.mount('/content/drive')

#os.chdir("drive/MyDrive/Thesis")
# Going to the folder that containes the data
os.chdir("PVDefectsDS/Update28Feb2023/")
directories = ["Serie1Update/", "Serie2/", "Serie3/", "Serie4/", "Serie5/"]

png_folder_path = "CellsGS/"
mat_folder_path = "MaskGT/"

# Load each PNG file as a numpy array and add it to a list
i = 0
png_arrays = []
fault_arrays = []

for j in range(len(directories)):
	os.chdir(directories[j])
	# Get a list of all PNG files in the folder
	png_file_paths = glob.glob(os.path.join(png_folder_path, "*.png"))

	# Load each PNG file as a numpy array and add it to a list
	for png_file_path in png_file_paths:
		with Image.open(png_file_path) as img:
			png_file_name = os.path.splitext(os.path.basename(png_file_path))[0]
			png_file_name = png_file_name.replace("ImageGS", "Image")
			mask_file_name = f"GT_{png_file_name}.mat"
			mask_file_path = os.path.join(mat_folder_path, mask_file_name)
		  
		  # Load the mask file as a numpy array, or create an artificial mask if the file does not exist
			if os.path.exists(mask_file_path):
				#print(i) #Uncomment this to find indentations which contained a failt in the cell for debugging
				img = np.array(img)
				if directories[j] == "Serie5/":
					img = np.rot90(img)
				for _ in range(4):
					fault_arrays.append(1) #Append 4 times when creating the data augmentation
				# Here we start appending the images
				png_arrays.append(img)
				#We append a left-right flip #mirrored
				png_arrays.append(np.fliplr(img))
				# We append an up-down flipped 
				png_arrays.append(np.flipud(img))
				# lastly we append an up-down left-right image
				png_arrays.append(np.flipud(np.fliplr(img)))
				i += 4 

			else:
				#If there is not mask that are connected to the image that means it 
				# doesn't have any faults and is therefore assigned a 0 and will not 
				# undergo any image augmentation. 
				fault_arrays.append(0)
				png_arrays.append(np.array(img))
				i += 1
	os.chdir("../")


#reshape the image so they are the same size 
common_size = (256, 256)
resized_png_arrays = []
i = 0
for png_array in png_arrays:
    png_img = Image.fromarray(png_array)
    resized_png_img = png_img.resize(common_size)
    resized_png_array = np.array(resized_png_img)
    resized_png_arrays.append(resized_png_array)


print(f'Amount of cells in the dateset: {len(fault_arrays)}')
print(f'Amount of cells with faults: {sum(fault_arrays)}')

X_train, X_test, y_train, y_test = train_test_split(resized_png_arrays, fault_arrays, test_size=0.2, random_state=42)
# Split train data into train and validation sets (75/25 split)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

height, width = 256, 256
X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
X_train = X_train.reshape((-1, height, width, 1))
X_test = X_test.reshape((-1, height, width, 1))
X_val = X_val.reshape((-1,height, width, 1))


y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

print(f"Amount of faulty cells in the training data: {sum(y_train)[1]}")
print(f"Amount of good cells in the training data: {sum(y_train)[0]}\n")
print(f"Amount of faulty cells in the testing data: {sum(y_test)[1]}")
print(f"Amount of good cells in the testing data: {sum(y_test)[0]}\n")
print(f"Amount of faulty cells in the tvalidation data: {sum(y_val)[1]}")
print(f"Amount of good cells in the tvalidation data: {sum(y_val)[0]}\n")


# We define the AlexNet model 
def AlexNet(input_shape, num_classes):
  model = keras.Sequential()
  model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), 
                          strides=(4, 4), activation="relu", 
                          input_shape=input_shape))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
  model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                          strides=(1, 1), activation="relu", 
                          padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                          strides=(1, 1), activation="relu", 
                          padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                          strides=(1, 1), activation="relu", 
                          padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), 
                          strides=(1, 1), activation="relu", 
                          padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(4096, activation="relu"))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes, activation="softmax")) # The two here controlls the amount of different classes. 
  return model


## We go out of 2 folder to where we wanna save things 
os.chdir("../")
os.chdir("../")
# Then we acess our results folder and save it there. 
os.chdir("Saved_Results/")


# We then do some inizializaytion for differnt training parameters. 
# Currently we are just training on 1 model and 1 parameter of epochs (20)
#epochs_list = [5,10,15,20,25,30,35,40,50]
epochs_list = [250]
report = []
# Here we copy the y_test so we can minipulate it but also reuse it
y_testAlexNet = y_test
y_testAlexNet = tf.math.argmax(y_testAlexNet, axis=1)
cm_list = []
for i in range(len(epochs_list)):
	model_AlexNet = AlexNet((256,256,1),2) # The input is the input shape and the amount of classes. 
	model_AlexNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	history = model_AlexNet.fit(X_train, y_train, validation_data=(X_val,y_val), epochs = epochs_list[i], batch_size = 32)
	
	y_pred = model_AlexNet.predict(X_test)
	y_predAlexNet = y_pred
	y_predAlexNet = tf.math.argmax(y_predAlexNet, axis=1)
	cm = confusion_matrix(y_testAlexNet,y_predAlexNet)
	print(cm)
	print(classification_report(y_testAlexNet, y_predAlexNet))
	cm_list.append(cm)
	report.append(classification_report(y_testAlexNet, y_predAlexNet))
	
	# We create a loss functin plot to see if it stabelizes 
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('AlexNet loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	#plt.show()
	plt.savefig('AlexNet_loss_test_epochs_'+str(epochs_list[i])+'.png')
	

# So we get an overview of all the raults found at the end, however we don't need it if we are only looking at 1 type of epoch
if len(epochs_list) > 1: 
	for i in range(len(epochs_list)):
		print(f"Amount of Epocs: {epochs_list[i]}")
		print(f"Classification report: \n {report[i]}")

## We go out of 2 folder to where we wanna save things 
#os.chdir("../")
#os.chdir("../")
# Then we acess our results folder and save it there. 
#os.chdir("Saved_Results/")



# We try to save the model again 
#filename = "Xception_firsttry.joblib"
#joblib.dump(xception, filename)

print("******************************")
print("******* End of program *******")
print("******************************")

