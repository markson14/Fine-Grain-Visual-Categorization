
import numpy as np
import time
import math
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model
import keras.applications
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from resnet152 import  ResNet152
from  keras.models import load_model


# Loading the training data
# train_dir = r'D:\ExternalPycharmProject\Inat\Aves_Small_SS1_Augmented_DC'
# val_dir = r'D:\ExternalPycharmProject\Aves_test\CV_0'
train_dir = r'D:\ExternalPycharmProject\100classes_train\CV_0'          #100classes
# train_dir = r'D:\ExternalPycharmProject\Aves_train\CV_0'
# val_dir = r'D:\ExternalPycharmProject\Aves_test\CV_0'
val_dir = r'D:\ExternalPycharmProject\100classes_test\CV_0'           #100classes

batch_size = 64
seed = 9
transformation_ratio = .05

# tensorboard = keras.callbacks.TensorBoard(log_dir='logs/1', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# Make sure to change this to the designated model
# Default configuration from
# https://keras.io/preprocessing/image/
train_datagen = ImageDataGenerator(
        rescale=1.,
        rotation_range=transformation_ratio,
        shear_range=transformation_ratio,
        zoom_range=transformation_ratio,
        cval=transformation_ratio,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

#########################################################################################
# base_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_shape=(224,224,3))
# base_model.summary()
# last_layer = base_model.get_layer('fc2').output
# # last_layer = Dropout(0.9, name="Dropout")(last_layer)
# out = Dense(train_generator.num_classes, activation='softmax', name='output')(last_layer)
# model = Model(inputs=base_model.input, outputs=out)
# model.summary()

# for layer in model.layers[:-1]:
#     layer.trainable = False
#
##########################################################################################

base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
x = base_model.output
# x = Flatten()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dropout(0.25, name='Dropout1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
# x = Dropout(0.5, name='Dropout2')(x)
predictions = Dense(train_generator.num_classes, activation='softmax', name='output')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()

epoch_num = 10
optimizer1 = Adam(lr=0.0001, decay=0.005)
optimizer2 = Adam(lr=0.00005, decay=0.005)

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='auto')
# train the model on the new data for a few epochs

num_train_samples = train_generator.samples
train_epoch_steps = math.ceil(num_train_samples / batch_size)
num_val_samples = validation_generator.samples
val_epoch_steps = math.ceil(num_val_samples / batch_size)
train_history = model.fit_generator(train_generator,
                    steps_per_epoch=train_epoch_steps,
                    epochs=epoch_num,
                    validation_data=validation_generator,
                    validation_steps=val_epoch_steps,
                    verbose=1
                    )

model.save('VGG16_basline_100c.h5')


np_all = np.array(train_history.history['acc'])
np_all = np.append(np_all, np.array(train_history.history['loss']), axis=0)
np_all = np.append(np_all, np.array(train_history.history['val_acc']), axis=0)
np_all = np.append(np_all, np.array(train_history.history['val_loss']), axis=0)
np.savetxt('VGG16_basline_100c.txt', np_all)

# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)


# model = load_model("transferlearingDC_dropout0.25_avg_480.h5")

# we chose to train the top 2 inception blocks, i.e. we will freeze

for layer in model.layers[:6]:
   layer.trainable = False
for layer in model.layers[6:]:
   layer.trainable = True

# model = load_model("VGG16FinetuningCNN_1024_Aug_DC_5.h5")

# optimizer1 = SGD(lr=0.00005, momentum=0.9, decay=1e-6)
optimizer2 = Adam(lr=0.00005, decay=0.005)
epoch_num = 10
earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='auto')

#Begin Model Traininga
model.compile(loss='categorical_crossentropy',optimizer=optimizer2,metrics=['accuracy'])
t=time.time()
#	t = now()
num_train_samples = train_generator.samples
train_epoch_steps = math.ceil(num_train_samples / batch_size)
num_val_samples = validation_generator.samples
val_epoch_steps = math.ceil(num_val_samples / batch_size)
train_history = model.fit_generator(train_generator,
                    steps_per_epoch=train_epoch_steps,
                    epochs=epoch_num,
                    validation_data=validation_generator,
                    validation_steps=val_epoch_steps,
                    )
print('Training time: %s mins' % round((time.time()-t)/60, 2))

model.save('VGG16_Finetuning3_avg_do0.25_480.h5')

np_all = np.array(train_history.history['acc'])
np_all = np.append(np_all, np.array(train_history.history['loss']), axis=0)
np_all = np.append(np_all, np.array(train_history.history['val_acc']), axis=0)
np_all = np.append(np_all, np.array(train_history.history['val_loss']), axis=0)
np.savetxt('VGG16_Finetuning3_avg_do0.25_480.txt', np_all)









