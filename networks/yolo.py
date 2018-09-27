from pydarknet import Detector, Image
import cv2

class YoloV3:
  def __init__(self, config):
    if 'weights' in config:
      self.weights = bytes(config['weights'], encoding='utf-8')
    else:
      raise KeyError('weights key is missing')
    if 'data' in config:
      self.data = bytes(config['data'], encoding='utf-8')
    else:
      raise KeyError('data key is missing')
    if 'cfg' in config:
      self.cfg = bytes(config['cfg'], encoding='utf-8')
    else:
      raise KeyError('cfg key is missing')
    print(self.data, self.cfg, self.weights)
    # Init the network
    self.net = Detector(self.cfg, self.weights, 0, self.data)

  def predict(self, frame):
    dark_frame = Image(frame)
    results = self.net.detect(dark_frame)
    del dark_frame

    return results
  
  def segment(self, frame, classes=[]):
    predictions = self.predict(frame)
    for prediction in predictions:
      classLabel, confidence, (x, y, w, h) = prediction
      x1, x2, y1, y2 = int(x-w/2), int(x+w/2), int(y-h/2), int(y+h/2) 
      segment = frame[y1: y2,x1: x2,:]
      yield classLabel, confidence, segment
