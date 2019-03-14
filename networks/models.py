from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (
    Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)


class Models:
  def __init__(self, nb_classes, model, seq_length,
               saved_model=None, features_length=2048):

    # Set defaults.
    self.seq_length = seq_length
    self.saved_model = saved_model
    self.nb_classes = nb_classes

    metrics = ['accuracy']
    # if self.nb_classes >= 10:
    #   metrics.append('top_k_categorical_accuracy')

    # Get the appropriate model.
    if self.saved_model is not None:
      print("Loading model %s" % self.saved_model)
      self.model = load_model(self.saved_model)
    elif model == 'opticalFLow-lstm':
      self.input_shape = (seq_length, features_length)
      self.model = self.lstm()
    elif model == 'lstm':
      print("Loading LSTM model.")
      self.input_shape = (seq_length, features_length)
      self.model = self.lstm()

    # Now compile the network.
    optimizer = Adam(lr=1e-4, decay=1e-5)
    self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                       metrics=metrics)
    print(self.model.summary())

  def lstm(self):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(LSTM(512, return_sequences=True,
                   input_shape=self.input_shape,
                   dropout=0.5))
    model.add(LSTM(128, dropout=0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(self.nb_classes, activation='softmax'))

    return model
