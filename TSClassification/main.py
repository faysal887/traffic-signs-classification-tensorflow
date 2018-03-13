import tensorflow as tf
import skimage
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random
import collections as clts
import pdb
from random import randint



def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))] # get the names of 62 classes folders names into a list
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


def showHist(labels):
    # Make a histogram with 62 bins of the `labels` data
    n=len(set(labels))
    plt.hist(labels, n)

    # Show the plot
    plt.show()


def showRandomImages(images, cMap=None):

    # Determine the (random) indexes of the images
    traffic_signs = [300, 2250, 1500, 2500]

    # Fill out the subplots with the random images and add shape, min and max values
    for i in range(len(traffic_signs)):
        plt.subplot(4, 1, i + 1)
        plt.axis('off')
        plt.imshow(images[traffic_signs[i]], cmap=cMap)
        plt.subplots_adjust(wspace=0.5)
        plt.title('shape: '+str(images[traffic_signs[i]].shape))

    plt.subplots_adjust(hspace=0.9)
    plt.show()


def showEachClassImage(images, labels):

    # Get the unique labels from 4575 labels
    unique_labels = set(labels)

    # Initialize the figure
    plt.figure(figsize=(15, 15))

    # Set a counter
    i = 1

    # For each unique label,
    for label in unique_labels:
        # You pick the first image for each label
        image = images[labels.index(label)]
        # Define 64 subplots
        plt.subplot(8, 8, i)
        # turn off axes lines and labels
        plt.axis('off')
        # Add a title to each subplot
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        # Add 1 to the counter
        i += 1
        # And you plot this first image
        plt.imshow(image)

    plt.subplots_adjust(hspace=0.6)
    plt.show()


def resizeImages(images, n, m):
    # Rescale the images in the `images` array
    images28 = [transform.resize(image, (n, m)) for image in images]
    return images28


def rgb2Gray(images28):
    # Convert `images28` to an array
    images28 = np.array(images28)

    images28_gray = np.array([rgb2gray(img) for img in images28])
    # Convert `images28` to grayscale
    #images28_gray = rgb2gray(images28)

    return images28_gray


def defineNN(images32, labels):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
    y = tf.placeholder(dtype=tf.int32, shape=[None])

    images_flat = tf.contrib.layers.flatten(x)

    # logits are the values that are to be used as input to the softmax
    # a single logit is thought as a tensor that will be mapped to the probability by the softmax function
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    lossFunc = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(lossFunc)

    # train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss) # accuracy=0.07
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    correct_pred = tf.argmax(logits, 1)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)

    # named tuples
    data = clts.namedtuple('data', ['x', 'y', 'images_flat', 'logits', 'train_op', 'accuracy', 'correct_pred', 'loss'])
    NN = data(x, y, images_flat, logits, train_op, accuracy, correct_pred, loss)

    return NN

def trainNN(NN, images28_gray, labels):
    # ### Running The Neural Network

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(401):
        print('EPOCH', i)
        # session.run evaluates tensors in fetches, values in feed_dict are for the corresponding input values
        _, accuracy_val = sess.run([NN.train_op, NN.accuracy], feed_dict={NN.x: images28_gray, NN.y: labels})
        if i % 10 == 0:
            print ("Accuracy: ", NN.accuracy)#NN.loss)
        print('DONE WITH EPOCH')

    return sess


def evaluateNN(NN, images28_gray, labels, sess):
    # Pick 10 random images
    sample_indexes = random.sample(range(len(images28_gray)), 10)
    sample_images = [images28_gray[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]

    # Run the "predicted_labels" op.
    predicted = sess.run([NN.correct_pred], feed_dict={NN.x: sample_images})[0]

    # Print the real and predicted labels
    print(sample_labels)
    print(list(predicted))

    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1 + i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                 fontsize=12, color=color)
        plt.imshow(sample_images[i])

    plt.show()


def testData(test_data_dir, NN, sess):
    # Load the test data
    test_images, test_labels = load_data(test_data_dir)

    # Transform the images to 28 by 28 pixels
    test_images28 = resizeImages(test_images, 28, 28)

    # Convert to grayscale
    test_images28_gray = rgb2Gray(test_images28)

    # Run predictions against the full test set.
    predicted = sess.run([NN.correct_pred], feed_dict={NN.x: test_images28_gray})[0]

    # Calculate correct matches
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

    # Calculate the accuracy
    accuracy = float(match_count) / len(test_labels)

    # Print the accuracy
    print("Accuracy: {:.3f}".format(accuracy))



def main():
    ROOT_PATH = "/home/faysal/Desktop/Other/DS & ML/Coursera-Machine Learning/TensorFlow/Datacamp_Tutorial"
    train_data_dir = os.path.join(ROOT_PATH, "TrafficSigns/Training")
    test_data_dir = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

    # images[4575(n,m,3)], labels=4575
    images, labels = load_data(train_data_dir)

    #showHist(labels)

    #showRandomImages(images)

    #showEachClassImage(images, labels)

    images28 = resizeImages(images, 28, 28)
    #showRandomImages(images28)

    images28_gray = rgb2Gray(images28)
    #showRandomImages(images28_gray, "gray")

    NN = defineNN(images28_gray, labels) ##

    sess=trainNN(NN, images28_gray, labels)

    evaluateNN(NN, images28_gray, labels,sess)

    testData(train_data_dir, NN,sess)

main();