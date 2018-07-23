import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.applications.imagenet_utils import preprocess_input
from resnet152 import Scale

path_finetuning = r'D:\ExternalPycharmProject\Finetunekernel\vgg16'


# model_path = path_finetuning + '\VGG16_Finetuning3_avg_do0.25.h5'
model_path = 'ResNet_Finetuning_4b27_DC_avg_480.h5'
model_val_dir = r'D:\ExternalPycharmProject\Aves_test\CV_0'
# The size used during training
# target_size = (224,224)
target_size = (480,480)
batch_size = 32
# Whether to normalize the confusion matrix or not
cf_norm = True

# Generated variable that might be quite heavy, load it only once
# model = load_model(model_path)

model = load_model(model_path, custom_objects={'Scale':Scale})

class_labels = os.listdir(model_val_dir)
labels_to_num = {c:i for i,c in enumerate(class_labels)}

def generate_true_labels(test_dir):
    """
    Generating the list of true label for each validation image in order
    """
    true_labels = []
    for i,subdir in enumerate(os.listdir(test_dir)):
        true_labels += [i]*len(os.listdir(os.path.join(test_dir,subdir)))
    return np.array(true_labels)

true_labels = generate_true_labels(model_val_dir)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# test_datagen = ImageDataGenerator(rescale=1.)


validation_generator = test_datagen.flow_from_directory(
        model_val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False) # Must not be shuffled so it can be compared with the true labels

# List array of prediction probability for each class
# This cell is slow, don't rerun it unless necessary
predictions = model.predict_generator(validation_generator)

# print(predictions)

# Best prediction on each image
y_pred = np.argmax(predictions, axis=-1)

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title, fontsize=20)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label', fontsize=16)
#     plt.xlabel('Predicted label', fontsize=16)
#
# cnf_matrix = confusion_matrix(true_labels, y_pred)
# # Plot non-normalized confusion matrix
# plt.figure(figsize=(15,10))
# title = 'Confusion matrix, with normalization' if cf_norm else 'Confusion matrix, without normalization'
# plot_confusion_matrix(cnf_matrix, classes=class_labels,
#                       normalize=cf_norm,
#                       title=title)
# plt.show()
#
# plt.figure(figsize=(15,10))
# plot_confusion_matrix(cnf_matrix, classes=class_labels,
#                       normalize=False,
#                       title='No normalization')
# plt.show()


##Precesion Recall F1 Report

def classifaction_report_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3] + lines[-2:-1]:
        row = {}
        row_data = line.split()
        row['class'] = ' '.join(row_data[:-4])
        row['precision'] = float(row_data[-4])
        row['recall'] = float(row_data[-3])
        row['f1_score'] = float(row_data[-2])
        row['support'] = int(row_data[-1])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.set_index(dataframe.columns[0], inplace=True)
    return dataframe

# Disable warning just for this
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    report = classification_report(true_labels, y_pred, target_names=class_labels)
    report_df = classifaction_report_df(report)
print(report)