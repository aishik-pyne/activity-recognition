import os
import random
import cv2
import pandas as pd
from .stream import frameCount, streamer

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
        trainSet = trainSet.append(
            {"file": _file, "class": activity, "frames": frameCount(_file)}, ignore_index=True)

    if cache:
      trainSet.to_csv(kth.cachePathMeta, index=False)

    return trainSet

  @staticmethod
  def splitIntoClip(kthPath, kthClipBasePath, clipSize):
    def dumpBuffer(path, buffer):
      for idx, img in enumerate(buffer):
        cv2.imwrite('{}.jpg'.format(os.path.join(path, str(idx).zfill(3))), img)
      print("dump at {}".format(path))
    
    # Create Clip Folder
    kthClipPath = os.path.join(kthClipBasePath, "kth{}/".format(clipSize))
    if not os.path.isdir(kthClipPath):
      os.mkdir(kthClipPath)

    for activity in kth.listActivities(kthPath):
      # Create folder if not exists
      activityPath = os.path.join(kthClipPath, activity + "/")
      if not os.path.isdir(activityPath):
        os.mkdir(activityPath)
        print("{} folder created".format(activityPath))

      # Extract images from vid
      activityFiles = kth.listFiles(kthPath, activity, abosultePath=True)

      clipNumber = 0
      print("CLIPRESET")
      for _f in activityFiles:
        # Load the frames
        buffer = [frame for frame in streamer(_f)]

        # Count the clip ranges
        noOfFrames = frameCount(_f)
        clipsCount = noOfFrames // clipSize
        for _clip in range(clipsCount):
          clipPath = os.path.join(
              kthClipPath, activity, "clip{}".format(clipNumber))
          clipNumber += 1
          if not os.path.isdir(clipPath + "/"):
            os.mkdir(clipPath + "/")
            print("{} folder created".format(clipPath))
          dumpBuffer(clipPath, buffer[_clip*clipSize:(_clip+1)*clipSize])
        del buffer


