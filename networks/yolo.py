from pydarknet import Detector, Image


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

    # Init the network
    self.net = Detector(self.cfg, self.weights, self.data)

  def predict(self, frame):
    dark_frame = Image(frame)
    results = self.net.detect(dark_frame)
    del dark_frame

    return dark_frame
