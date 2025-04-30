# libs:
import numpy as np
import os
import tensorflow as tf
import pickle as pkl
import math
import sys
import argparse
from sklearn.svm import SVC

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# scripts:
import facenet
import detect_face

with tf.Graph().as_default():

    with tf.session() as sess:
        data_dir = './' # path to align face data
        d_set = facenet.get_dataset(data_dir)
        paths, labels = facenet.get_image_paths_and_labels(d_set)
        print('Number of classes: %d' % len(d_set))
        print('Number of images: %d' % len(paths))

        print('Loading feature extraction model')
        model_dir = './' # path to pre-trained model
        facenet.load_model(model_dir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # run forward pass to calculate the embeddings:
        print('Calculating features for images')
        batch_size = 1000
        image_size = 160
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
        
        classifier_filename = './' # path to save the classifier model (./my_classifier.pkl)
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        # training classfier:
        print('Traingin the classifier..')
        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in d_set]

        # save classifier model:
        with open (classifier_filename_exp, 'wb') as outfile:
            pkl.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
        print('Goodluck')

    