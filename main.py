import json
import os
import time
import cv2
from colored import fg, bg, attr
from utils import streamer, frameCount
from utils import kth
from networks import YoloV3 


config = json.load(open("config.json", "r"))
kth_dataset_path = config["paths"]["kth_dataset"]

def test_kth():
    print("{}Testing kth...{}".format(bg(1), attr('reset')))
    # List activities
    activities = kth.listActivities(kth_dataset_path)
    files = {}
    for activity in activities:
        files[activity] = kth.listFiles(kth_dataset_path, activity=activity, abosultePath=True)
        print("activity: {} \t count: {}".format(activity, len(files[activity])))
        
def test_yolo():
    print("{}Testing yolo...{}".format(bg(1), attr('reset')))
    net = YoloV3(config["paths"]["yolov3"])
    print("Frames in the file {}".format(frameCount(kth.randomFile(kth_dataset_path))))
    i =0 
    for frame in streamer(kth.randomFile(kth_dataset_path)):
        i+=1
        # print(frame)
        net.segment(frame)
        if i==10:
            break
test_kth()
test_yolo()