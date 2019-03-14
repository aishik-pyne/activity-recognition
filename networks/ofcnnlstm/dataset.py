import threading
import os
import pandas as pd
import numpy as np
import cv2
from keras.utils import to_categorical

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


class DataSet:

  def __init__(self, base_path, seq_length=30, class_limit=None, image_shape=(224, 224, 3)):
    self.data_file = pd.read_csv(os.path.join(base_path, 'data_file.csv'))
    self.classes = self.get_classes()

  def get_classes(self):
    """
    Gets the list of classes
    """
    return sorted(list(set(self.data_file["class"])))

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
      img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
      if shape:
        img = cv2.resize(img, shape)
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

  def build_image_sequence(self, sample):
    """
    :param sample: A dict having keys 'clip', 'class'
    """
    if os.path.isdir(sample['clip']):
      image_paths = sorted([os.path.join(sample['clip'], x)
                            for x in os.listdir(sample['clip'])])
      x = np.array([self.load_image(p) for p in image_paths])
      y = np.array([self.get_class_one_hot(sample['class'])
                                           for _ in image_paths])
      return (x, y)
    else:
      raise ValueError("sample['clip'] is not a valid directory")

  def split_train_test(self):
    """Split the data into train and test groups."""
    train = self.data_file[self.data_file['type'] == 'train']
    test = self.data_file[self.data_file['type'] == 'test']
    return train, test

  @threadsafe_generator
  def clip_batch_generator(self, batch_size, train_test, data_type=None):
    """Return a generator that we can use to train on. There are
    a couple different things we can return:
    data_type: 'features', 'images'
    """
    # Get the right dataset for the generator.
    train, test = self.split_train_test()
    data = train if train_test == 'train' else test
    print("Creating %s generator with %d samples." %
          (train_test, len(data)))

    while 1:
      X, y = [], []

      samples = data.sample(batch_size).to_dict('records')
      yield [self.build_image_sequence(s) for s in samples]


if __name__ == "__main__":
  ds = DataSet('/data/datasets/kth_clip/features')
  print(ds.classes)
  gen = ds.clip_batch_generator(batch_size=16, train_test='train')
  batch = next(gen)
  x, y = batch[0]
  pprint(type(x))
  pprint(x.shape)
  pprint(type(y))
  pprint(y.shape)
  pprint(y)
