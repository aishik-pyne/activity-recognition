import random
import cv2
import json
from tqdm import tqdm
from pprint import pprint
from collections import Counter
from keras.models import model_from_json
from utils import kth


def model_accuracy(model, data_root, sample_size=10):
  # Create test set
  activities = kth.listActivities(data_root)
  test_set = []
  for activity in activities:
    files = kth.listFiles(data_root, activity, abosultePath=True)
    test_files = random.sample(files, sample_size)
    [test_set.append((tst_f, activity)) for tst_f in test_files]

  # Measure accuracy
  correct_pred = 0
  for test_img, test_gt in tqdm(test_set, desc="Test Case"):
    cap = cv2.VideoCapture(test_img)
    predicitons = Counter()
    while cap.isOpened():
      ret, frame = cap.read()
      if ret:
        pred = model.predict(frame)
        predicitons[pred] += 1
      else:
        break
    most_common_pred = predicitons.most_common()

    if test_gt == most_common_pred:
      correct_pred += 1

  # Return accuracy
  return correct_pred / len(test_set)


if __name__ == "__main__":
  data_root = '/data/datasets/kth_dataset/'
  with json.load('weights/') as f:
    model = model_from_json(f)
  print(model_accuracy(model, data_root))
