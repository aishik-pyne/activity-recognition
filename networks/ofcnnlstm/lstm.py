from keras import Sequential
from keras.layers import LSTM, Dropout, Dense


def lstm(nb_classes):
  model = Sequential()
  model.add(LSTM(512, return_sequences=True,
                 dropout=0.5))
  # model.add(LSTM(128, dropout=0.5, ))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes, activation='softmax'))

  return model