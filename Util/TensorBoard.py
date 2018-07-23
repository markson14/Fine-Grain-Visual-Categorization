

#This is not a program sample, it is more like a step-by-step tutorial

#1 put the below script into you training class

tensorboard = keras.callbacks.TensorBoard(log_dir='logs/1', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)\


#2 remember the callbacks in fit function, for example:
train_history = custom_vgg_model.fit_generator(train_generator,
                    steps_per_epoch=train_epoch_steps,
                    epochs=epoch_num,
                    validation_data=validation_generator,
                    validation_steps=val_epoch_steps,callbacks=[tensorboard])

#3 run cmd command

tensorboard --logdir="the path you create in log_dir, it will created automatically under the same folder of your class"

#4 follow the cmd to open localhost:6006 and it will lead you to the tensorboard

#for more information refers to https://keras.io/callbacks/