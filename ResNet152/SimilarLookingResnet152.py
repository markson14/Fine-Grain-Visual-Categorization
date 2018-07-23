
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
from resnet152 import Scale
from keras.applications.imagenet_utils import preprocess_input
# Loading the training data
# train_dir = r'D:\ExternalPycharmProject\Inat\Aves_Small_SS1_Augmented_DC'
# train_dir = r'D:\ExternalPycharmProject\Inat\Aves_Small_SS1_Augmented_SC'
train_dir = r'D:\ExternalPycharmProject\100classes_train\CV_0'          #100classes
# train_dir = r'D:\ExternalPycharmProject\Aves_train\CV_0'
# val_dir = r'D:\ExternalPycharmProject\Aves_test\CV_0'
val_dir = r'D:\ExternalPycharmProject\100classes_test\CV_0'           #100classes

batch_size = 64       # for 224 input size
# batch_size = 22         # for 480 input size
seed = 9
transformation_ratio = .05

# tensorboard = keras.callbacks.TensorBoard(log_dir='logs/1', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# Make sure to change this to the designated model
# Default configuration from
# https://keras.io/preprocessing/image/
train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=transformation_ratio,
        shear_range=transformation_ratio,
        zoom_range=transformation_ratio,
        cval=transformation_ratio,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(480, 480),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(480, 480),
        batch_size=batch_size,
        class_mode='categorical')

#########################################################################################

# base_model.summary()
# last_layer = base_model.get_layer('fc2').output
# # last_layer = Dropout(0.9, name="Dropout")(last_layer)
# out = Dense(train_generator.num_classes, activation='softmax', name='output')(last_layer)
# model = Model(image_input, out)
# model.summary()
#
# for layer in model.layers[:-1]:
#     layer.trainable = False
#
# model.layers[6].trainable

base_model = ResNet152(include_top=False, weights='imagenet', input_shape=(480,480,3), pooling='avg')
x = base_model.output
# x = Flatten()(x)
x = Dense(1024, activation='relu', name='fc')(x)
predictions = Dense(train_generator.num_classes, activation='softmax', name='output')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# model.summary()


epoch_num = 5
optimizer = Adam(lr=0.001, decay=0.005)
optimizer1 = Adam(lr=0.0001, decay=0.005)
optimizer2 = Adam(lr=0.00005, decay=0.005)

model.compile(optimizer=optimizer1, loss='categorical_crossentropy',metrics=['accuracy'])
# earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='auto')
# train the model on the new data for a few epochs

t=time.time()
num_train_samples = train_generator.samples
train_epoch_steps = math.ceil(num_train_samples / batch_size)
num_val_samples = validation_generator.samples
val_epoch_steps = math.ceil(num_val_samples / batch_size)
train_history = model.fit_generator(train_generator,
                    steps_per_epoch=train_epoch_steps,
                    epochs=epoch_num,
                    validation_data=validation_generator,
                    validation_steps=val_epoch_steps,
                    verbose=2
                    )

model.save('transferleanring_res_avg_480.h5')
# np_all = np.array(train_history.history['acc'])
# np_all = np.append(np_all, np.array(train_history.history['loss']), axis=0)
# np_all = np.append(np_all, np.array(train_history.history['val_acc']), axis=0)
# np_all = np.append(np_all, np.array(train_history.history['val_loss']), axis=0)
# np.savetxt('Resnet_baseline_SC_avg.txt', np_all)

t1 = time.time()
timespent=t1 - t
print('Training time: %s mins' % round(timespent/60, 2))





for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
model = load_model(r"D:\ExternalPycharmProject\TransferLearningMiddleKernel\resnet\transferleanring_res_dc_480_avg.h5", custom_objects={'Scale':Scale})
for layer in model.layers[:547]:
   layer.trainable = False
for layer in model.layers[547:]:
   layer.trainable = True


# optimizer1 = SGD(lr=0.00005, momentum=0.9, decay=1e-6)
optimizer2 = Adam(lr=0.00005, decay=0.005)
optimizer3 = Adam(lr=0.00002, decay=0.005) # for 480 fine tuning
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
                    verbose=1)

t1 = time.time()
timespent=t1 - t
print('Training time: %s mins' % round(timespent/60, 2))


model.save('ResNet_Finetuning_4b27_DC_avg_480.h5')

np_all = np.array(train_history.history['acc'])
np_all = np.append(np_all, np.array(train_history.history['loss']), axis=0)
np_all = np.append(np_all, np.array(train_history.history['val_acc']), axis=0)
np_all = np.append(np_all, np.array(train_history.history['val_loss']), axis=0)
np.savetxt('ResNet_Finetuning_4b27_DC_avg_480.txt', np_all)











