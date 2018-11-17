from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Flatten
class LstmModels:
  def __init__(self, baseModel='vgg16'):
    if baseModel == 'vgg16':
      self.baseModel = VGG16(include_top=False, weights='imagenet', input_shape=(120, 160, 3), classes=6)
    

  def extractClipFeatures(self, images):
    pass

  def cnnLstm(self):
    model = Sequential()

LstmModels()