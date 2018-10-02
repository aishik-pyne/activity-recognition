import os
import random
import pandas as pd
from .stream import frameCount


class kth:
  cachePathMeta = "/tmp/kth_dataset_meta.csv"
  @staticmethod
  def listActivities(path):
    '''
      - path: kth dataset path
    '''
    return os.listdir(path)

  @staticmethod
  def listFiles(path, activity=None, abosultePath=False):
    '''
      - path: kth dataset path
      - activity: name of the activity. defaults to a random activity
    '''
    if not activity:
      activity = random.choice(kth.listActivities(path))

    files = os.listdir(os.path.join(path, activity))
    aviFiles = list(filter(lambda x: x.endswith(".avi"), files))

    if abosultePath:
      aviFiles = [os.path.join(path, activity, _file) for _file in aviFiles]

    return aviFiles

  @staticmethod
  def randomFile(path, activity=None):
    return random.choice(kth.listFiles(path, activity=activity, abosultePath=True))

  @staticmethod
  def metadata(path, cache=True):
    if cache:
      
      if os.path.isfile(kth.cachePathMeta):
        return pd.read_csv(kth.cachePathMeta, index_col=False)

    trainSet = pd.DataFrame(columns=["class", "file", "frames"])
    activities = kth.listActivities(path)
    for activity in activities:
      files = kth.listFiles(path, activity, True)
      for _file in files:
        trainSet = trainSet.append({"file": _file, "class": activity, "frames": frameCount(_file)}, ignore_index=True)

    if cache:
      trainSet.to_csv(kth.cachePathMeta, index=False)

    return trainSet
