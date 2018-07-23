import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='', type=str,
                    help='The parent directory of images to be completed.')
parser.add_argument('--mask_dir', default='', type=str,
                    help='The parent directory of mask, value 255 indicates mask.')
parser.add_argument('--output_dir', default='', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


restart_model = False

if __name__ == "__main__":
    #ng.get_gpus(1)
    args = parser.parse_args()
    model = InpaintCAModel()
    # Let's just create the background of limited number of masks
    start_index = 115
    end_index = 120
    print('index',start_index,end_index)
    # Looping from the mask directory as not all original images have mask
    for subdir_name in os.listdir(args.mask_dir):
        print('Working on',subdir_name)
        mask_subdir_path = os.path.join(args.mask_dir, subdir_name)
        # Create the subdir for the output if it does not exist
        output_subdir_path = os.path.join(args.output_dir, subdir_name)
        if os.path.isdir(output_subdir_path) == False:
            os.mkdir(output_subdir_path)
        for mask_name in os.listdir(mask_subdir_path)[start_index:end_index]:
            if restart_model == True:
                print('restarting the model')
                model = InpaintCAModel()
                restart_model = False
            mask_image_path = os.path.join(mask_subdir_path, mask_name)
            image_path = os.path.join(args.image_dir, subdir_name, mask_name)
            #print(mask_image_path)
            #print(image_path)
            mask = cv2.imread(mask_image_path)
            # Image path should be mirroring the image path
            image = cv2.imread(image_path)

            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            #print('Shape of image: {}'.format(image.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            with tf.Session(config=sess_config) as sess:
                input_image = tf.constant(input_image, dtype=tf.float32)
                output = model.build_server_graph(input_image, reuse=tf.AUTO_REUSE)
                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)
                # load pretrained model
                vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
                    assign_ops.append(tf.assign(var, var_value))
                # In case of the GraphDef cannot be larger than 2GB error, quick hotfix for now
                try:
                    sess.run(assign_ops)
                except Exception as e:
                    print('error when running session',e)
                    # Need to restart the model as well
                    restart_model = True
                    continue
                #print('Model loaded.')
                result = sess.run(output)
                output_image_path = os.path.join(output_subdir_path, mask_name)
                cv2.imwrite(output_image_path, result[0][:, :, ::-1])
