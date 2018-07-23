import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
import cv2
# Configurable parameter
model_path = r'D:\ExternalPycharmProject\Finetunekernel\VGG16Finetuning_1024_Aug_SC.h5'
# Just to get the label name from the class number
model_train_dir = r'D:\ExternalPycharmProject\Inat\Aves_Small_SS1_Augmented_DC'
# Last convolutional layer name, search it first using model.summary()
last_conv_layer_name = 'block5_conv3'
# Dimension of images used to train the model (after resizing)
train_model_width = 224
train_model_height = 224

# Generated variable that might be quite heavy, load it only once
model = load_model(model_path)
class_labels = os.listdir(model_train_dir)
class_to_label = {c:i for i,c in enumerate(class_labels)}


# The main function
# Taken from the Deep Learning with Python textbook, with a little bit of modification to generalize it
def generate_prediction_heatmap(model, class_labels,
                                train_model_width, train_model_height,
                                last_conv_layer_name,
                                input_image_path,
                                output_image_dir,
                                heatmap_class_label=-1,
                                skip_true_prediction=False,
                                verbose=True):
    if verbose:
        print('Loading image from:', input_image_path)
    img = image.load_img(input_image_path, target_size=(train_model_height, train_model_width))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)


    y_prob = model.predict(x)
    best_y_class = y_prob.argmax(axis=-1)[0]
    if verbose:
        print('-' * 20)
        for i, prob in enumerate(y_prob[0]):
            print('Class:', class_labels[i], ', probability:', '{:.4f}'.format(prob))
        print('-' * 20)
        print('Predicted :', class_labels[best_y_class], 'as the best class')

    # If generating the heatmap of true prediction is not necessary, i.e. we are only interested in comparing false prediction
    if skip_true_prediction and best_y_class == heatmap_class_label:
        if verbose:
            print('Skipping ', input_image_path)
        return

    # Also use the supplied label to generate the heatmap if supplied, use only the best class otherwise
    y_classes = [best_y_class] if heatmap_class_label == -1 else [best_y_class, heatmap_class_label]

    # Generate all the heatmap image of pre-determined list of classes
    for y_class in y_classes:
        out = model.output[:, y_class]
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grads = K.gradients(out, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.Function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        for i in range(last_conv_layer.output_shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        img = cv2.imread(input_image_path)
        # Need to be converted due to weird colour ordering of the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        # abcd1234_classname.jpg seems like a better format
        output_image_name = os.path.basename(input_image_path).split('.')[0] + '_' + class_labels[y_class] + '.jpg'
        output_image_path = os.path.join(output_image_dir, output_image_name)
        if verbose:
            print('Generating output heatmap image at:', output_image_path)
            print()
        _ = cv2.imwrite(output_image_path, superimposed_img)
        del superimposed_img, heatmap, img

# the directory version of main script
# Configurable directory parameter
species_name = 'Setophaga ruticilla'
# Whether to also generate the true class image or not, otherwise only generate the best predicted class
use_true_species_label = True
# In case that we are only interested in generating false predicted images, usually for validation images
skip_true_prediction = True
# Keep this one low (but >0) if true prediction is generated
image_limit = -1
# input_image_base_dir = r'D:\Resources\Inat_Partial\Aves_Small_SS2_Train\CV_0'
# output_image_base_dir = r'D:\Dummy\temp_out\heatmap_ss2_train'
input_image_base_dir = r'D:\ExternalPycharmProject\Aves_test\CV_0'
output_image_base_dir = 'HeatMapSC'

# ---------------------------------------------------------------------------
heatmap_class_label = class_to_label[species_name]  if use_true_species_label else False
input_image_dir = os.path.join(input_image_base_dir,species_name)
output_image_dir = os.path.join(output_image_base_dir,species_name)
if os.path.isdir(output_image_dir) == False:
    os.mkdir(output_image_dir)
for input_image_name in os.listdir(input_image_dir)[:image_limit]:
    input_image_path = os.path.join(input_image_dir,input_image_name)
    generate_prediction_heatmap(model, class_labels,
                                train_model_width, train_model_height,
                                last_conv_layer_name,
                                input_image_path, output_image_dir,
                                heatmap_class_label,
                                skip_true_prediction=skip_true_prediction,
                                verbose=True)