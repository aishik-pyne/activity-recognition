from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
                                        MaxPooling2D)
import tensorflow as tf


class LSTMModel():
    def __init__(self, nb_classes, seq_length, features_length,
                 saved_model=None):
        self.nb_classes = nb_classes
        self.seq_length = seq_length
        self.saved_model = saved_model
        self.features_length = features_length

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Load model
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.input_shape = (seq_length, features_length)
            self.model = load_model(self.saved_model)
        else:
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()

        # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        # Compile model
        optimizer = Adam(lr=1e-4, decay=1e-5)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(1024, return_sequences=True,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(LSTM(64, return_sequences=False,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
