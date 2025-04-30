# libs:
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import random
from scipy import misc
from time import sleep

# scripts:
import facenet
import detect_face

# path to output folder:
out_dir_path = './'
out_dir = os.path.expanduser(out_dir_path)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# path to input image folder:
data_dir = './'
d_set = facenet.get_dataset(data_dir)

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUoptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './') # path to det1.npy

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
image_size = 182

# adding a random key to the filename to allow alignment using multiple process.
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(out_dir, 'bounding_boxes_%05d.txt' % random_key)
print("Goodluck...")

with open(bounding_boxes_filename, "w") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in d_set:
        out_class_dir = os.path.join(out_dir, cls.name)
        if not os.path.exists(out_class_dir):
            os.makedirs(out_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            out_filename = os.path.join(out_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(out_filename):
                try:
                    img = misc.imread(image_path)
                    print('read data dimension: ', img.ndim)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (out_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                        print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
                    print('after data dimension: ', img.ndim)

                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('detected_faces: %d' % nrof_faces)
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                            img_center = img_size / 2
                            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],(det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                            det = det[index, :]
                        det = np.squeeze(det)
                        bb_temp = np.zeros(4, detype=np.int32)

                        bb_temp[0] = det[0]
                        bb_temp[1] = det[1]
                        bb_temp[2] = det[2]
                        bb_temp[3] = det[3]

                        cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                        scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')

                        nrof_successfully_aligned += 1
                        misc.imsave(out_filename, scaled_temp)
                    else:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (out_filename))

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
