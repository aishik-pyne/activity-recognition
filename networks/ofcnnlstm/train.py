from keras import Sequential, Model
from keras.models import model_from_json
from keras import layers
from keras.applications import vgg16

import cv2
import os
import numpy as np
from opticalflow import dense_opticalflow_to_hsl
from cnn import vgg16_flattened
from dataset import DataSet
from lstm import lstm

IMG_WIDTH, IMG_HEIGHT = 120, 160

BASE_CNN = vgg16_flattened()
DATASET = DataSet('/data/datasets/kth_clip/features')

def preprocessing(sequence):
  global BASE_CNN
  opflow = dense_opticalflow_to_hsl(sequence)
  feature = BASE_CNN.predict(opflow)

  return feature


def train(model: Sequential, data_generator) -> Sequential:
  
  model.fit_generator(data_generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
  return model


def export(model, export_path):
  pass

def load(export_path):

if __name__ == "__main__":
  # #   graph = build_graph()
  # sequence_path = "/data/datasets/kth_clip/kth30/boxing/clip1/"
  # sequence = np.array([cv2.imread(os.path.join(sequence_path, file),
  #                                 cv2.IMREAD_COLOR) for file in sorted(os.listdir(sequence_path))])

  print(next(DATASET.clip_batch_generator(1, 'train')))

