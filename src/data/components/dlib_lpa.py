from torch.utils.data import Dataset
import gdown
import zipfile
from tqdm import tqdm
import os
import typing
from typing import Optional
import albumentations as A
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

class DLIB_LPA(Dataset):
  def __init__(self, transform: Optional[A.Compose] = None):
      self.data_dir = None
      self.img_labels = None
      self.transform = transform
      self.url = 'https://drive.google.com/file/d/1JK2-1GKnL2dJ7rQMxq1RZrbPpl7klfs9/view?usp=sharing'
      self.id = '1JK2-1GKnL2dJ7rQMxq1RZrbPpl7klfs9'
      self.file_name = '300WLPA_2d.zip'
      self.origin_path = 'data/300W'
      self.folder_dir = '300WLPA_2d'
      self.labels_files = ['300WLPA_AFW_1.txt', '300WLPA_HELEN_1.txt', '300WLPA_HELEN_10001.txt', '300WLPA_HELEN_20001.txt', '300WLPA_HELEN_30001.txt', '300WLPA_LFPW.txt']
      if not os.path.exists(self.origin_path):
          os.makedirs(self.origin_path)
          
      self.prepare_data()
      self.prepare_labels()

  def download(self):
      gdown.download(url=self.url, output=os.path.join(self.origin_path, self.file_name), quiet=False, use_cookies=False, fuzzy=True) #id=self.id,

  def unzip(self):
      file_name = os.path.join(self.origin_path, self.file_name)
      with zipfile.ZipFile(file_name, mode='r', allowZip64=True) as file:
          members = file.infolist()
          progress = tqdm(members)
          for member in progress:
              try:
                  file.extract(member, self.origin_path)
                  progress.set_description(f"Extracting {member.filename}")
              except zipfile.error as e:
                  pass
      file.close()

  def prepare_data(self):
      if not os.path.isfile(os.path.join(self.origin_path, self.file_name)):
          self.download()
      for labels_file in self.labels_files:
          if not os.path.isfile(os.path.join(self.origin_path, self.folder_dir, labels_file)):
              self.unzip()
      if os.path.isdir(os.path.join(self.origin_path, self.folder_dir)):
          self.data_dir = os.path.join(self.origin_path, self.folder_dir)

  def prepare_labels(self):
      if self.data_dir is None:
          self.prepare_data()
    
      if self.img_labels is None:
          data = []
          for labels_file in self.labels_files:
              labels_dir = os.path.join(self.data_dir, labels_file)
              with open(labels_dir, 'r') as file:
                  lines = file.readlines()
                  for line in lines:
                      values = line.split()
                      file_name = values[0]
                      row = {'File name': file_name}
                      points = list(map(float, values[1:]))
                      for i in range(68):
                          row.update({
                              f"Point {i + 1}": (points[i], points[i+68])
                              })
                      data.append(row)
          df = pd.DataFrame(data)
          if not df.empty:
              self.img_labels = df

  def __len__(self):
      if self.img_labels is None:
          self.prepare_labels()
      return len(self.img_labels)

  def parse_roi_box_from_landmark(self, pts: typing.List[typing.Tuple[int, int]])->typing.List[int]:
      """calc roi box from landmark"""
      x = [pt[0] for pt in pts]  # x-coordinates
      y = [pt[1] for pt in pts]  # y-coordinates
      bbox = [min(x), min(y), max(x), max(y)]
      center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
      radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
      bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
      
      llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
      center_x = (bbox[2] + bbox[0]) / 2
      center_y = (bbox[3] + bbox[1]) / 2
      
      roi_box = [0] * 4
      roi_box[0] = center_x - llength / 2
      roi_box[1] = center_y - llength / 2
      roi_box[2] = roi_box[0] + llength
      roi_box[3] = roi_box[1] + llength
      
      return roi_box

  def __getitem__(self, index):
      if index < 0 or index >= len(self):
          raise IndexError(f"Index {index} is out of range")

      if self.data_dir is None:
          self.prepare_data()
      if self.img_labels is None:
          self.prepare_labels()

      if self.data_dir is not None and self.img_labels is not None:
          img_path = os.path.join(self.data_dir, self.img_labels.iloc[index, 0])
          landmark = self.img_labels.iloc[index, -68:]
          # keypoints = np.array(landmark)
          image = Image.open(img_path).convert('RGB')
          roi_box = self.parse_roi_box_from_landmark(landmark)
          area = (roi_box[0], roi_box[1], roi_box[2], roi_box[3])
          image = image.crop(area)
          keypoints = []
          for i in range(68):
              x = landmark[i][0] - roi_box[0]
              y = landmark[i][1] - roi_box[1]
              keypoints.append((x, y))
          keypoints = np.array(keypoints)
          sample = {'image': image, 'landmark': keypoints, 'box': roi_box} #, 'box': roi_box
      return sample

  @staticmethod
  def visual_keypoints(image: Image, keypoints: np.ndarray) -> None:
      plt.imshow(image)
      x_values = [x for x, y in keypoints]
      y_values = [y for x, y in keypoints]
      plt.scatter(x_values, y_values, s=10, marker='.', c='r')
      plt.axis('off')
      plt.savefig('test.png')
      plt.show()

  @staticmethod
  def image_annotation(image: Image, keypoints: np.ndarray)->Image:
      draw = ImageDraw.Draw(image)
      for i in range(keypoints.shape[0]):
          draw.ellipse(xy=(keypoints[i][0]-1, keypoints[i][1]-1, keypoints[i][0]+1, keypoints[i][1]+1), fill=(0, 255, 0))
      return image