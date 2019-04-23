import os
import pandas as pd
import numpy as np
from subprocess import call
from tqdm import tqdm
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from cnn import vgg16_flattened
KTH_PATH = '/data/datasets/kth_dataset/'
KTH_IMG = '/data/datasets/kth_img/'
KTH_FEATURE = '/data/datasets/kth_feature/'
CLASSES = ['boxing', 'handclapping',
           'handwaving', 'jogging', 'running', 'walking']


def get_video_list():
  videos = pd.DataFrame(
      columns=['path', 'class', 'img_folder', 'feature_folder'])
  for c in CLASSES:
    for f in os.listdir(os.path.join(KTH_PATH, c)):
      abs_path = os.path.join(KTH_PATH, c, f)
      img_folder = os.path.join(KTH_IMG, os.path.splitext(f)[0])
      feature_folder = os.path.join(KTH_FEATURE, os.path.splitext(f)[0])
      videos = videos.append({
          'path': abs_path,
          'class': c,
          'img_folder': img_folder,
          'feature_folder': feature_folder
      }, ignore_index=True)
  return videos


def test_train_split(df, split=0.8):
  df = df.sample(frac=1)
  msk = np.random.rand(len(df)) < split
  return df[msk], df[~msk]


def vid_to_img(videos):
  """Convert a video into all it's frame

  :param videos: Dataframe of video file paths
  :type videos: list
  """
  global KTH_IMG
  with tqdm(total=len(list(videos.iterrows()))) as pbar:
    for idx, (src, klass, img_folder) in videos.iterrows():
      print(img_folder)
      if not os.path.isdir(img_folder):
        os.mkdir(img_folder)
        dest = os.path.join(img_folder, 'image-%04d.jpg')
        print(dest)
        call(["ffmpeg", "-i", src, dest])
      pbar.update(1)


def process_image(image, target_shape):
  """Given an image, process it and return the array."""
  # Load the image.
  h, w, _ = target_shape
  image = load_img(image, target_size=(h, w))

  # Turn it into numpy, normalize and return.
  img_arr = img_to_array(image)
  x = (img_arr / 255.).astype(np.float32)

  return np.array(x)


def img_to_feature(videos):
  global KTH_IMG
  cnn = vgg16_flattened(image_shape=(160, 120, 3))
  with tqdm(total=len(list(videos.iterrows()))) as pbar:
    for idx, (src, klass, img_folder, feature_folder) in videos.iterrows():
      # Process each frame in the img folder
      if not os.path.isdir(feature_folder):
        os.mkdir(feature_folder)
      frames = np.array([process_image(image=os.path.join(
          img_folder, img), target_shape=(160, 120, 3)) for img in os.listdir(img_folder)])
      features = cnn.predict(frames)
      np.save(os.path.join(feature_folder, 'feature.npy'), features)
      pbar.update(1)
if __name__ == "__main__":
  v = get_video_list()

  # Get CSV File dumps
  # train, test = test_train_split(v)
  # train.to_csv('./data/train.csv', index=False)
  # test.to_csv('./data/test.csv', index=False)

  # Video to image
  # vid_to_img(v)

  # Image to features
  img_to_feature(v)
