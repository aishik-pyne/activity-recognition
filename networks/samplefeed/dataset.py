import threading
import os
import numpy as np
import cv2
import csv
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from random import sample
from pprint import pprint


class threadsafe_iterator:
    """
    Thread safe iterator
    """

    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen


CLASSES = ['boxing', 'handclapping',
           'handwaving', 'jogging', 'running', 'walking']


class ImageDataSet:

    def __init__(self, seq_length=30, image_shape=(160, 120, 3), mode='train'):
        """
        :seq_length : Vid of n frames is subsampled to seq_length frames
        :image_shape : Image shape
        :mode : Dataset mode train/test
        """
        self.seq_length = seq_length
        self.image_shape = image_shape
        self.mode = mode

        # self.data = self.get_data()
        # self.classes = self.get_classes()

    @property
    def data(self):
        """Load our data from file."""
        with open(os.path.join('data', '{}.csv'.format(self.mode)), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data[1:]

    @property
    def classes(self):
        """
        Gets the list of classes
        """
        return sorted(list(set([c[1] for c in self.data])))

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def load_image(self, abs_path, shape=None):
        """
        Loads a image and returns a np array from absolute filepath
        """
        if os.path.isfile(abs_path):
            img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
            # img = cv2.resize(img, self.image_shape)
            return img
        else:
            raise ValueError("File doesn't exist {}".format(abs_path))

    def load_video(self, abs_path, shape=None):
        """
        Loads a video and returns a series np arrays from absolute filepath
        """
        if os.path.isfile(abs_path):
            frames = []
            cap = cv2.VideoCapture(abs_path)
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    break
                if shape:
                    img = cv2.resize(img, shape)
                frames += img
            return np.array(frames)
        else:
            raise ValueError("File doesn't exist")

    def build_image_sequence(self, img_folder):
        """Reads frames from a video file

        :param img_folder: Folder where frames of the video is stored
        :type img_folder: str
        :raises ValueError: Invalid path
        :return: list of frames
        :rtype: np.array
        """
        if os.path.isdir(img_folder):
            image_paths = sorted([os.path.join(img_folder, x)
                                  for x in os.listdir(img_folder)])
            # frames = [self.load_image(p) for p in image_paths]
            frames = self.rescale_list(image_paths, self.seq_length)

            return [self.process_image(x, self.image_shape) for x in frames]
        else:
            raise ValueError(
                "Folder {} is not a valid directory".format(img_folder))

    def rescale_list(self, input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list.

        :param input_list: [description]
        :type input_list: [type]
        :param size: size of rescaled list
        :type size: int
        :return: [description]
        :rtype: [type]
        """

        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def process_image(self, image, target_shape):
        """Given an image, process it and return the array."""
        # Load the image.
        h, w, _ = target_shape
        image = load_img(image, target_size=(h, w))

        # Turn it into numpy, normalize and return.
        img_arr = img_to_array(image)
        x = (img_arr / 255.).astype(np.float32)

        return x

    @threadsafe_generator
    def frame_generator(self, batch_size):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        :param batch_size: An int
        """
        print("Creating %s generator with %d samples." %
              (self.mode, len(self.data)))

        while 1:
            batch = sample(self.data, batch_size)
            X, Y = [], []
            for b in batch:
                X.append(self.build_image_sequence(b[2]))
                Y.append(self.get_class_one_hot(b[1]))

            # samples = self.data.sample(batch_size).to_dict('records')
            # yield [self.build_image_sequence(s) for s in samples]
            yield np.array(X), np.array(Y)


class VGGFeatureDataSet():

    def __init__(self, seq_length=30, mode='train'):
        """
        :seq_length : Vid of n frames is subsampled to seq_length frames
        :mode : Dataset mode train/test
        """
        self.seq_length = seq_length
        self.mode = mode

        # self.data = self.get_data()
        # self.classes = self.get_classes()

    @property
    def data(self):
        """Load our data from file."""
        with open(os.path.join('data', '{}.csv'.format(self.mode)), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data[1:]

    @property
    def classes(self):
        """
        Gets the list of classes
        """
        return sorted(list(set([c[1] for c in self.data])))

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def rescale_list(self, input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list.

        :param input_list: [description]
        :type input_list: [type]
        :param size: size of rescaled list
        :type size: int
        :return: [description]
        :rtype: [type]
        """

        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def build_feature_sequence(self, feature_folder):
        """Reads frames from a video file

        :param feature_folder: Folder where vgg features of video is stored 
        :type feature_folder: str
        :raises ValueError: Invalid path
        :return: list of frames
        :rtype: np.array
        """
        if os.path.isdir(feature_folder):
            feature = np.load(os.path.join(feature_folder, 'feature.npy'))
            return self.rescale_list(feature, self.seq_length)
            image_paths = sorted([os.path.join(feature_folder, x)
                                  for x in os.listdir(feature_folder)])
            # frames = [self.load_image(p) for p in image_paths]
            frames = self.rescale_list(image_paths, self.seq_length)

            return [self.process_image(x, self.image_shape) for x in frames]
        else:
            raise ValueError(
                "Folder {} is not a valid directory".format(feature_folder))

    @threadsafe_generator
    def frame_generator(self, batch_size):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        :param batch_size: An int
        """
        print("Creating VGGFeature %s generator with %d samples." %
              (self.mode, len(self.data)))

        while 1:
            batch = sample(self.data, batch_size)
            X, Y = [], []
            for b in batch:
                X.append(self.build_feature_sequence(b[3]))
                Y.append(self.get_class_one_hot(b[1]))

            # samples = self.data.sample(batch_size).to_dict('records')
            # yield [self.build_image_sequence(s) for s in samples]
            yield np.array(X), np.array(Y)

if __name__ == "__main__":
    ds = ImageDataSet(seq_length=40)
    # fg = ds.frame_generator(1)

    vgg_ds = VGGFeatureDataSet(seq_length=40)
    fg = vgg_ds.frame_generator(1)
    pprint(next(fg)[0].shape)
