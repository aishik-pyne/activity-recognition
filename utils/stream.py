import cv2, os

def frameCount(filePath):
    if not os.path.isfile(filePath):
        raise ValueError("filePath isn't an actual path")
    vidcap = cv2.VideoCapture(filePath)
    return int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
def streamer(path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    while success:
        yield image
        success,image = vidcap.read()