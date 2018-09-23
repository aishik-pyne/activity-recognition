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
    # net = YoloV3(config["paths"]["yolov3"])
    print("Frames in the file {}".format(frameCount(kth.randomFile(kth_dataset_path))))
    for frame in streamer(kth.randomFile(kth_dataset_path)):
        pass
test_kth()
test_yolo()
# if __name__ == "__main__":
#     # Optional statement to configure preferred GPU. Available only in GPU version.
#     # pydarknet.set_cuda_device(0)

#     net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
#                    bytes("cfg/coco.data", encoding="utf-8"))

#     cap = cv2.VideoCapture(0)

#     while True:
#         r, frame = cap.read()
#         if r:
#             start_time = time.time()

#             # Only measure the time taken by YOLO and API Call overhead

#             dark_frame = Image(frame)
#             results = net.detect(dark_frame)
#             del dark_frame

#             end_time = time.time()
#             print("Elapsed Time:",end_time-start_time)

#             for cat, score, bounds in results:
#                 x, y, w, h = bounds
#                 cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
#                 cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

#             cv2.imshow("preview", frame)

#         k = cv2.waitKey(1)
#         if k == 0xFF & ord("q"):
#             break