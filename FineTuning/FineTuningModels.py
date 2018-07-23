
import numpy as np
import os
import time

from keras.models import load_model

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
import keras.applications
from keras.callbacks import Callback
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import pickle as pickle
from keras.optimizers import Adam
from keras.optimizers import SGD
from resnet152 import ResNet152

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = r'D:\ExternalPycharmProject\Aves'
data_dir_list = os.listdir(data_path)

img_data_list=[]


# Icon = 0

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		# img = image.load_img(img_path, target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		# x = x/255
		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 10
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:222]=0
labels[222:522]=1
labels[522:868]=2
labels[868:1084]=3
labels[1084:1330]=4
labels[1330:1631]=5
labels[1631:1931]=6
labels[1931:2290]=7
labels[2290:2651]=8
labels[2651:2906]=9

names=['Setophaga americana', 'Setophaga coronata', 'Setophaga coronata auduboni', 'Setophaga coronata coronata',
       'Setophaga magnolia', 'Setophaga palmarum', 'Setophaga petechia', 'Setophaga ruticilla', 'Setophaga townsendi',
       'Setophaga virens']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
#########################################################################################
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))
# image_input = Input(shape=(299, 299, 3))
base_model = ResNet152(include_top=False, weights='imagenet', input_tensor=image_input, classes=1000)
# base_model = keras.applications.DenseNet201(include_top=False, weights='imagenet', input_tensor=image_input, classes=1000)
# base_model = keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=image_input, classes=1000)
x = base_model.output
# x = Flatten()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', name='fc1')(x)
# x = Dropout(0.2, name='Dropout1')(x)
# x = Dense(1024, activation='relu', name='fc2')(x)
# x = Dropout(0.5, name='Dropout2')(x)
x = Dense(num_classes, activation='softmax', name='output')(x)
model = Model(image_input, x)

for layer in base_model.layers:
    layer.trainable = False

model.summary()

optimizer1 = Adam(lr=0.0001, decay=0.05)

model.compile(optimizer=optimizer1, loss='categorical_crossentropy', metrics=['accuracy'])
t=time.time()
# train the model on the new data for a few epochs
hist = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=2, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))




for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:673]:
   layer.trainable = False
for layer in model.layers[673:]:
   layer.trainable = True


# model = load_model("DenseNet201_Finetuning_2.h5")


optimizer = Adam(lr=0.00005, decay=0.01)

# earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=0, verbose=0, mode='auto')

#Begin Model Traininga
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
t=time.time()
#	t = now()
hist = model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=2, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))




model.save('Resnet152_Finetuning1.h5')

np_all = np.array(hist.history['acc'])
np_all = np.append(np_all, np.array(hist.history['loss']), axis=0)
np_all = np.append(np_all, np.array(hist.history['val_acc']), axis=0)
np_all = np.append(np_all, np.array(hist.history['val_loss']), axis=0)
np.savetxt('Resnet152_Finetuning11.txt', np_all)





