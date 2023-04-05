print("******************************")
print("**** Beginning og program ****")
print("******************************")


# Resnet 152 for solar panel cells including image augmenation
print("Resnet 152 for solar panel cells including image augmenation with ")
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
				if directories[j] == "Serie5/": # Since series5 have photos that are all ortaed 90 degrees to the other series we rotate there images before performing the image augmentation 
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

#f, ax = plt.subplots(2,2, figsize =(8,8))
#ax[0,0].imshow(png_arrays[10])
#ax[0,0].set_title('Image')
#ax[0,1].imshow(png_arrays[11])
#ax[0,1].set_title('Mirrored')
#ax[1,0].imshow(png_arrays[12])
#ax[1,0].set_title('Flipped')
#ax[1,1].imshow(png_arrays[13])
#ax[1,1].set_title('Flipped and mirrored')

#print(fault_arrays[10],fault_arrays[11],fault_arrays[12],fault_arrays[13], fault_arrays[14] )

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



ROWS = 256
COLS = 256
CHANNELS = 1
CLASSES = 2

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    
    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet152(input_shape = (256, 256, 1), classes = 2):   
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Initial stage - Is in the in all ResNet models 
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 1, block='a', s = 1) #x1 stage conv2_x
    X = identity_block(X, 3, [64, 64, 256], stage=1, block='b') # x2 stage conv2_x
    X = identity_block(X, 3, [64, 64, 256], stage=1, block='c') # x3 stage conv2_x

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 2, block='a', s = 2) # Will add a convolutional block for resizing and will be x1 stage conv3_x
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='b') # x2 stage conv3_x
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='c') # x3 stage conv3_x
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='d') # x4 stage conv3_x
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='e') # x5 stage conv3_x
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='f') # x6 stage conv3_x
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='g') # x7 stage conv3_x
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='h') # x8 stage conv3_x

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 3, block='a', s = 2) #x1 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='b') #x2 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='c') #x3 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='d') #x4 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='e') #x5 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='f') #x6 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='g') #x7 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='h') #x8 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='i') #x9 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='j') #x10 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='k') #x11 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='l') #x12 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='m') #x13 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='n') #x14 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='o') #x15 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='p') #x16 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='q') #x17 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='r') #x18 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='s') #x19 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='t') #x20 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='u') #x21 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='v') #x22 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='w') #x23 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='x') #x24 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='y') #x25 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='z') #x26 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='aa') #x27 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='bb') #x28 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='cc') #x29 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='dd') #x30 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='ee') #x31 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='ff') #x32 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='gg') #x33 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='hh') #x34 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='ii') #x35 conv4_x
    X = identity_block(X, 3, [256, 256, 1024], stage=3, block='jj') #x36 conv4_x


    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=4, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=4, block='c')

    # AVGPOOL.
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet152')

    return model


#epochs_list = [5,10,15,20,25,30,35,40,50]
epochs_list = [250]
report = []

y_test_resnet152 = y_test
y_test_resnet152 = tf.math.argmax(y_test_resnet152, axis=1)
cm_list = []
for i in range(len(epochs_list)):
	model_resnet152 = ResNet152(input_shape = (ROWS, COLS, CHANNELS), classes = CLASSES)
	model_resnet152.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	history = model_resnet152.fit(X_train, y_train, validation_data=(X_val,y_val), epochs = epochs_list[i], batch_size = 32)

	y_pred = model_resnet152.predict(X_test)
	y_pred_resnet152 = y_pred
	y_pred_resnet152 = tf.math.argmax(y_pred_resnet152, axis=1)
	
	# Evaluation parameters
	cm_resnet152 = confusion_matrix(y_test_resnet152,y_pred_resnet152)
	print(cm_resnet152)
	cm_list.append(cm_resnet152)
	
	print(classification_report(y_test_resnet152, y_pred_resnet152))
	report.append(classification_report(y_test_resnet152, y_pred_resnet152))
	
	# Create an image for the loss funciton to see if and when it stabalizes. 
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Resnet152 loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	#plt.show() #since this code is run in the HPC we can't show() images so we have to save them instead. 
	plt.savefig('resnet152_loss_test_epochs_'+str(epochs_list[i])+'.png')

# So we get an overview of all the raults found at the end, however we don't need it if we are only looking at 1 type of epoch
if len(epochs_list) > 1: 
	for i in range(len(epochs_list)):
		print(f"Amount of Epocs: {epochs_list[i]}")
		print(f"Classification report: \n {report[i]}")

print("******************************")
print("******* End of program *******")
print("******************************")
