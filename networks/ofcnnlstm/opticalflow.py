import cv2
import os
import numpy as np


def dense_opticalflow_to_hsl(image_list):
  """
  Calculates dense optical flow of a sequence of images and returns the hsl
  :param image_list: np array of shape (<seq_length>, <width>, <height>, 3)
  :return flow_list: np array of shape (<seq_length>, <width>, <height>, 3)
  """
  _, width, height, channels = image_list.shape
  flow_list = []
  prev_frame = np.zeros((width, height))

  hsv = np.zeros((width, height, channels), dtype=np.uint8)
  hsv[..., 1] = 255

  for image in image_list:
    next_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_list.append(bgr)

    prev_frame = next_frame

  return np.array(flow_list)


if __name__ == "__main__":
  """
  Testing the optical flow for a clip
  """
  sequence_path = "/data/datasets/kth_clip/kth30/boxing/clip1/"
  sequence = np.array([cv2.imread(os.path.join(sequence_path, file), cv2.IMREAD_COLOR)
              for file in sorted(os.listdir(sequence_path))])
  # sequence = np.expand_dims(sequence, axis=-1)
  opflow = dense_opticalflow_to_hsl(sequence)
  
  for f in opflow:
    cv2.imshow('frame2',f)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
      break

  cv2.destroyAllWindows()