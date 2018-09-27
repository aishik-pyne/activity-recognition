import os
import random


class kth:
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
