import tensorflow as tf
import numpy as np
import random
import gzip
import tarfile
import pickle
import os
from six.moves import urllib
from plot import *

class data_pipeline:
    def __init__(self,type):
        self.type = type
        self.debug = 0
        self.batch = 0

        if self.type == "MNIST":
            self.url = "http://yann.lecun.com/exdb/mnist/"
            self.debug =1
            self.n_train_images = 60000
            self.n_test_images = 10000
            self.n_channels = 1
            self.size = 28
            self.MNIST_filename = ["train-images-idx3-ubyte.gz",
                              "train-labels-idx1-ubyte.gz",
                              "t10k-images-idx3-ubyte.gz",
                              "t10k-labels-idx1-ubyte.gz"]

        elif self.type == "CIFAR_10":
            self.url = "https://www.cs.toronto.edu/~kriz/"
            self.debug = 1
            self.n_train_images = 50000
            self.n_test_images = 10000
            self.n_channels = 3
            self.size = 32
            self.CIFAR_10_filename = ["cifar-10-python.tar.gz"]

        assert self.debug == 1, "Data type must be MNIST or CIFAR_10"

    def maybe_download(self, filename, filepath):
        if os.path.isfile(filepath) is True:
            print("Filename %s is already downloaded" % filename)
        else:
            filepath,_ = urllib.request.urlretrieve(self.url + filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
            print("Successfully download", filename, size, "bytes")
        return filepath


    def download_data(self):
        self.filepath_holder = []

        if not tf.gfile.Exists("./Data"):
            tf.gfile.MakeDirs("./Data")

        if self.type == "MNIST":
            for i in self.MNIST_filename:
                filepath = os.path.join("./Data", i)
                self.maybe_download(i,filepath)
                self.filepath_holder.append(filepath)

        elif self.type == "CIFAR_10":
            for i in self.CIFAR_10_filename:
                filepath = os.path.join("./Data", i)
                self.maybe_download(i,filepath)
                self.filepath_holder.append(filepath)
        print("-" * 80)

    def extract_mnist_images(self, filepath, size, n_images,n_channels):
        print("Extracting and Reading ", filepath)

        with gzip.open(filepath) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(size*size*n_images*n_channels)
            data = np.frombuffer(buf, dtype = np.uint8)
            data = np.reshape(data,[n_images, size, size, n_channels])
        return data

    def extract_mnist_labels(self, filepath,n_images):
        print("Extracting and Reading ", filepath)

        with gzip.open(filepath) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1*n_images)
            labels = np.frombuffer(buf, dtype = np.uint8)
            one_hot_encoding = np.zeros((n_images, 10))
            one_hot_encoding[np.arange(n_images), labels] = 1
            one_hot_encoding = np.reshape(one_hot_encoding, [-1,10])
        return one_hot_encoding

    def extract_cifar_data(self,filepath, train_files,n_images):
        ## this code is from https://github.com/melodyguan/enas/blob/master/src/cifar10/data_utils.py
        images, labels = [], []
        for file_name in train_files:
            full_name = os.path.join(filepath, file_name)
            with open(full_name, mode = "rb") as finp:
                data = pickle.load(finp, encoding = "bytes")
                batch_images = data[b'data']
                batch_labels = np.array(data[b'labels'])
                images.append(batch_images)
                labels.append(batch_labels)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        one_hot_encoding = np.zeros((n_images, 10))
        one_hot_encoding[np.arange(n_images), labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, 10])
        images = np.reshape(images, [-1, 3, 32, 32])
        images = np.transpose(images, [0, 2, 3, 1])

        return images, one_hot_encoding

    def extract_cifar_data_(self,filepath, num_valids=5000):
        print("Reading data")
        with tarfile.open(filepath, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, "./Data")
        images, labels = {}, {}
        train_files = [
            "./cifar-10-batches-py/data_batch_1",
            "./cifar-10-batches-py/data_batch_2",
            "./cifar-10-batches-py/data_batch_3",
            "./cifar-10-batches-py/data_batch_4",
            "./cifar-10-batches-py/data_batch_5"]
        test_file = ["./cifar-10-batches-py/test_batch"]
        images["train"], labels["train"] = self.extract_cifar_data("./Data", train_files,self.n_train_images)

        if num_valids:
            images["valid"] = images["train"][-num_valids:]
            labels["valid"] = labels["train"][-num_valids:]

            images["train"] = images["train"][:-num_valids]
            labels["train"] = labels["train"][:-num_valids]
        else:
            images["valid"], labels["valid"] = None, None

        images["test"], labels["test"] = self.extract_cifar_data("./Data", test_file,self.n_test_images)
        return images, labels

    def apply_preprocessing(self, images, mode):
        mean = np.mean(images, axis =(0,1,2))
        images = images/255
        print("%s_mean: " % mode, mean)
        return images

    def load_preprocess_data(self):
        self.download_data()
        if self.type == "MNIST":
            train_images = self.extract_mnist_images(self.filepath_holder[0],self.size, self.n_train_images, self.n_channels)
            train_labels = self.extract_mnist_labels(self.filepath_holder[1], self.n_train_images)
            self.valid_images = train_images[0:5000,:,:,:]
            self.valid_labels = train_labels[0:5000,:]
            self.train_images = train_images[5000:,:,:,:]
            self.train_labels = train_labels[5000:,:]
            self.test_images = self.extract_mnist_images(self.filepath_holder[2],self.size, self.n_test_images, self.n_channels)
            self.test_labels = self.extract_mnist_labels(self.filepath_holder[3], self.n_test_images)
            print("-" * 80)
            self.train_images = self.apply_preprocessing(images = self.train_images, mode = "train")
            self.valid_images = self.apply_preprocessing(images = self.valid_images, mode = "valid")
            self.test_images = self.apply_preprocessing(images = self.test_images, mode = "test")
            print("-" * 80)
            print("training size: ", np.shape(self.train_images),", ",np.shape(self.train_labels))
            print("valid size:    ", np.shape(self.valid_images), ", ", np.shape(self.valid_labels))
            print("test size:     ", np.shape(self.test_images), ", ", np.shape(self.test_labels))
        else:
            images, labels = self.extract_cifar_data_(self.filepath_holder[0])
            self.train_images = images["train"]
            self.train_labels = labels["train"]
            self.valid_images = images["valid"]
            self.valid_labels = labels["valid"]
            self.test_images = images["test"]
            self.test_labels = labels["test"]
            print("-" * 80)
            self.train_images = self.apply_preprocessing(images = self.train_images, mode = "train")
            self.valid_images = self.apply_preprocessing(images = self.valid_images, mode = "valid")
            self.test_images = self.apply_preprocessing(images = self.test_images, mode = "test")
            print("-" * 80)
            print("training size: ", np.shape(self.train_images),", ",np.shape(self.train_labels))
            print("valid size:    ", np.shape(self.valid_images), ", ", np.shape(self.valid_labels))
            print("test size:     ", np.shape(self.test_images), ", ", np.shape(self.test_labels))

        return self.train_images, self.train_labels, self.valid_images, self.valid_labels, self.test_images, self.test_labels

    def make_noise(self,image):

        def gaussian_noise(image):
            size = np.shape(image)
            noise = np.random.normal(0,0.3, size = size)
            image = image + noise

            return image

        return gaussian_noise(image)

    def initialize_batch(self):
        self.batch = 0

    def next_batch(self, images, labels, batch_size, make_noise = None):

        if make_noise is False:
            self.length = len(images)//batch_size
            batch_xs = images[self.batch*batch_size: self.batch*batch_size + batch_size,:,:,:]
            batch_noised_xs = np.copy(batch_xs)
            batch_ys = labels[self.batch*batch_size: self.batch*batch_size + batch_size,:]
            self.batch += 1
            if self.batch == (self.length):
                self.batch = 0
        else:
            self.length = len(images)//batch_size
            batch_noised_xs = []
            batch_xs = images[self.batch*batch_size: self.batch*batch_size + batch_size,:,:,:]
            batch_ys = labels[self.batch * batch_size: self.batch * batch_size + batch_size, :]

            if self.type == "MNIST":
                _ = np.reshape(batch_xs, [-1, self.size, self.size])
                for i in range(batch_size):
                    batch_noised_xs.append(self.make_noise(_[i]))
                batch_noised_xs = np.reshape(batch_noised_xs, [-1, self.size, self.size, self.n_channels])
            else:
                for i in range(batch_size):
                    batch_noised_xs.append(self.make_noise(batch_xs[i]))

            self.batch += 1
            if self.batch == (self.length):
                self.batch = 0

        return batch_xs, batch_noised_xs, batch_ys

    def get_total_batch(self,images, batch_size):
        self.batch_size = batch_size
        return len(images)//self.batch_size
