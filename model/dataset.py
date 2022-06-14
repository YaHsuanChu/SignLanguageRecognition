import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset



class asl_alphabet_dataset_keypoints(Dataset):

  def __init__(self, dataPath, kpClassifier, size=2500):
    self.images ,self.labels = readData( dataPath, small = True, size=size ) #numpy array
    self.kpClassifier = kpClassifier
    landmarks = self.kpClassifier.images_to_landmarks( self.images )
    keypoints = self.kpClassifier.landmarks_to_float_array(landmarks)
    self.data = torch.FloatTensor( keypoints.reshape( (keypoints.shape[0], keypoints.shape[1]*keypoints.shape[2]) ) )
    self.labels = torch.FloatTensor( self.labels )
    self.empty_label = torch.FloatTensor( np.zeros( (29,) ,dtype = np.int32 ) ) #when a hand is not detected
    self.empty_label[27] = 1

  def __getitem__(self, index):
    
    if self.data[index].any(): #exist a hand
        return self.data[index], self.labels[index]
    else: #no hands detected
        return self.data[index], self.empty_label
        
  def __len__(self):
    return len(self.data)

class asl_alphabet_dataset(Dataset):

  def __init__(self, dataPath, size=500):
    self.images, self.labels = readData( dataPath, small = True, size=size) #numpy array
    
  def __getitem__(self, index):
    return self.images[index], self.labels[index]
  
  def __len__(self):
    return len(self.images)


def string_to_label( label_str ):
  
  '''
  label_str: the name of each subdirectories 
  (29 classes: 'A'-'Z', 'del', 'nothing' ,'space')
  label: numpy array of shape (29,) one-hot encoding label vector
  '''

  if len( label_str ) == 1:
    pos = ord( label_str.upper() ) - ord('A')
  else:
    if label_str == 'del':
      pos = 26
    elif label_str == 'nothing':
      pos = 27
    elif label_str == 'space':
      pos = 28
    else:
      raise Exception( 'Error: class not defined' )
  
  label = np.zeros( (29,) ,dtype = np.int32 )
  label[pos] = 1
  
  return label


def int_to_string( label_int ):
    
    if label_int == 26: return 'del'
    elif label_int == 27: return 'nothing'
    elif label_int == 28: return 'space'
    else:
        return str( chr( ord('a')+label_int ) )
        
def readData( dataPath, small = False, size = 50 ):
  
  '''
  return 
  images: numpy array of shape( N, 200, 200, 3 )
  labels: numpy array of shape( N, 29 )
  '''

  images = []
  labels = []
  subdirs = os.listdir( dataPath )  #usually returns 29 subdir

  for subdir in subdirs:
    imagelist = os.listdir( os.path.join( dataPath, subdir ) )
    label = string_to_label( subdir ) #numpy array of shape (29,)_
    
    if small: imagelist = imagelist[0: min( size, len(imagelist) )]
    
    for image in imagelist:
      img = cv2.imread( os.path.join( dataPath, subdir, image ) )
      images.append( img )
      labels.append( label )
    
    print( 'There are ', len(imagelist), ' ', subdir, ' images' )
  
  print( len( images ), ' images in ', dataPath, ' in total' )

  return np.array( images ), np.array( labels )
  
def readDataResize( dataPath, x, y, small = False, size = 50 ):
  
  '''
  return 
  images: numpy array of shape( N, x, y, 3 )
  labels: numpy array of shape( N, 29 )
  '''

  images = []
  labels = []
  subdirs = os.listdir( dataPath )  #usually returns 29 subdir

  for subdir in subdirs:
    imagelist = os.listdir( os.path.join( dataPath, subdir ) )
    label = string_to_label( subdir ) #numpy array of shape (29,)_
    
    if small: imagelist = imagelist[0: min( size, len(imagelist) )]

    for image in imagelist:
      img = cv2.imread( os.path.join( dataPath, subdir, image ) )
      img = cv2.resize(img, (x, y), cv2.INTER_AREA)
      images.append( img )
      labels.append( label )
    
    print( 'There are ', len(imagelist), ' ', subdir, ' images' )
  
  print( len( images ), ' images in ', dataPath, ' in total' )

  return np.array( images ), np.array( labels )
