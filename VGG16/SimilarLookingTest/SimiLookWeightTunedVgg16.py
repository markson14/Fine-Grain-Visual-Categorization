
import numpy as np
import os
import time
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
import keras.applications
from keras.models import load_model

from sklearn.cross_validation import train_test_split

from keras import backend as K
from sklearn.utils import class_weight




Icon = 0

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '\Bird'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		x = x/255
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

labels[0:17]=0
labels[17:783]=1
labels[783:893]=2
labels[893:915]=3
labels[915:937]=4
labels[937:1050]=5
labels[1050:1079]=6
labels[1079:1094]=7
labels[1094:1120]=8
labels[1120:1489]=9


names = ['2746','2747','2748','2749','2750','2751','2752','2753','2754','2755']
class_weight  = {0:21.70, 1:0.48, 2:3.35, 3: 16.77, 4:16.77, 5:3.26, 6:12.72, 7:24.6, 8:14.19, 9:1}

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#########################################################################################
#Training the classifier alone

if(Icon == 0):
	image_input = Input(shape=(224, 224, 3))

	model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=image_input, classes=1000)
	model.summary()
	last_layer = model.get_layer('fc2').output
	#x= Flatten(name='flatten')(last_layer)
	out = Dense(num_classes, activation='softmax', name='output')(last_layer)
	custom_vgg_model = Model(image_input, out)
	custom_vgg_model.summary()

	for layer in custom_vgg_model.layers[:-1]:
		layer.trainable = False

	custom_vgg_model.layers[6].trainable


elif(Icon == 1):
	custom_vgg_model = load_model('VGG16custom_WeightChange_Simi.h5')
#Begin Model Training
custom_vgg_model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
t=time.time()
#	t = now()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=16, epochs=24, verbose=2, validation_data=(X_test, y_test), class_weight=class_weight)
print('Training time: %s' % (t - time.time()))




(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

custom_vgg_model.save('VGG16custom_WeightChange_Simi.h5')

np_all = np.array(hist.history['acc'])
np_all = np.append(np_all, np.array(hist.history['loss']), axis=0)
np_all = np.append(np_all, np.array(hist.history['val_acc']), axis=0)
np_all = np.append(np_all, np.array(hist.history['val_loss']), axis=0)
np.savetxt('save.txt', np_all)






