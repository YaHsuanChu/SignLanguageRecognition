#keypointsModel.py
'''
this file defines 2 models with each correspond to a class repectively:
1. ASL_alphabet_kp_fc_classifier( TRAIN_PATH, TEST_PATH ) 
2. ASL_alphabet_kp_nb_classifier( TRAIN_PATH, TEST_PATH )
Besides, the class keypointsClassifier is also defined in this file  
'''

import cv2
import mediapipe
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
import torch
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader

from .dataset import readData
from .dataset import asl_alphabet_dataset_keypoints
from .dataset import asl_alphabet_dataset
from .util import analyse_confusion_matrix

'''========== class keypointsClassifiers =========='''
class keypointsClassifier:

  def __init__(self, handThresh=0.2, keypointThresh=0.2): #since there are some false negative when detection hands, I set handTHresh to 0.2

    self.mp_hands = mediapipe.solutions.hands.Hands( static_image_mode=True,
                              max_num_hands=1,
                              min_detection_confidence=handThresh,
                              min_tracking_confidence=keypointThresh )
    self.mp_drawing = mediapipe.solutions.drawing_utils
    self.mp_drawing_style = mediapipe.solutions.drawing_styles

  def loadData(self, dataPath): #if we're using Dataset object than we don't need this function

    return readData( dataPath, small = True, size = 1500 )
  
  def images_to_landmarks(self, images):

    '''
    images: shape(N, 200, 200, 3) BGR image
    keypoints: (N, landmarks, 21 landmark, xyz) value between [0,1]
    '''

    keypoints = []
    for image in images:

      result = self.mp_hands.process( cv2.cvtColor( image, cv2.COLOR_BGR2RGB ) )
      keypoints.append( result )
    
    return np.array( keypoints )

  def landmarks_to_float_array(self, landmarks):
    '''
    convert landmark object into numpy array
    return all zero if no hands are detected in an image

    landmarks: array of detection results
    Hierachy of landmarks:
    1. N: number of images
    2. .multi_hand_world_landmarks
    3. for each hand
    4. .landmark
    5. landmark[ 0:20 ]
    6. landmark[k].x(y)(z)

    arr: numpy array of shape (N, 21, 3) , type: float, [0,1]
    '''
    arr = []
    
    for n in range( len(landmarks) ):
      if landmarks[n].multi_hand_landmarks is not None:
        for hand in landmarks[n].multi_hand_world_landmarks:
          single = np.array( [np.array( [ hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z ] )for i in range(21) ] )
          normalized = (single-single.mean(axis=0))/single.std(axis=0)
          arr.append( normalized )
          break #only 1 hand is allowed
      else:
        arr.append( np.zeros((21, 3), dtype=np.float32 ))

    return np.array( arr )

  def visualize_keypoints(self, images, keypoints, save_dir):

    save = True
    description = 'try'
    number = 0

    new_images = images.copy()

    for i in range( len(new_images) ):
      if keypoints[i].multi_hand_landmarks is not None:
        for hand in keypoints[i].multi_hand_landmarks:
          self.mp_drawing.draw_landmarks( new_images[i], hand, mediapipe.solutions.hands.HAND_CONNECTIONS, self.mp_drawing_style.get_default_hand_landmarks_style(), self.mp_drawing_style.get_default_hand_connections_style()  )
          
      if save:
        cv2.imwrite( save_dir+description+str(number)+'.jpg', new_images[i])
        print( 'save an image: ', save_dir+description+str(number)+'.jpg' )
      number = number+1
      
    return new_images 


'''========== class fc_classifier_net =========='''
class fc_classifier_net(torch.nn.Module):

  def __init__(self, nInput, nHidden1, nHidden2, nHidden3, nOut):
    super( fc_classifier_net, self ).__init__()
    self.hiddenLayer1 = torch.nn.Sequential(
              torch.nn.Linear( nInput, nHidden1 ),
              torch.nn.ReLU()
    )
    self.hiddenLayer2 = torch.nn.Sequential(
              torch.nn.Linear( nHidden1, nHidden2 ),
              torch.nn.ReLU()
    )
    self.hiddenLayer3 = torch.nn.Sequential(
              torch.nn.Linear( nHidden2, nHidden3 ),
              torch.nn.ReLU()
    )
    self.outLayer = torch.nn.Sequential(
              torch.nn.Linear( nHidden3, nOut ),
              torch.nn.Softmax( dim=1 )
    )
  def forward(self, x):
    x = self.hiddenLayer1(x)
    x = self.hiddenLayer2(x)
    x = self.hiddenLayer3(x)
    x = self.outLayer(x)
    return x

'''========== class ASL_alphabet_kp_fc_classifier =========='''
class ASL_alphabet_kp_fc_classifier:
  '''a class that make use of keypointsClassifier + fc_classifier to perform training, testing and inference'''
  
  def __init__(self, TRAIN_PATH=None, TEST_PATH=None ):

    self.TRAIN_PATH = TRAIN_PATH 
    self.TEST_PATH = TEST_PATH

    self.kpClassifier = keypointsClassifier()
    self.fc_net = fc_classifier_net(  63, 25, 20, 20, 29 ) #(nInput, hidden1, hidden2, nOut) 
    self.LR = 0.05
    self.epochs = 1000
    self.batch_size = 64

    self.trainSet = None #class: torch.utils.data.Dataset
    self.testSet = None #class: torch.utils.data.Dataset

  def load_fc_model(self, model_path, map_location='cuda' ):
    self.fc_net.load_state_dict( torch.load(model_path,map_location=torch.device(map_location) )  )

  def train(self, size=2500, save_path='./model/fc_cls_unnamed.pt'):

    #check device
    if torch.cuda.is_available(): 
      device = 'cuda'
      self.fc_net.cuda()
    else: device = 'cpu'

    #initialize dataset (compute keypoints here)
    self.trainSet = asl_alphabet_dataset_keypoints( self.TRAIN_PATH, self.kpClassifier, size ) #class: torch.utils.data.Dataset
    trainLoader = DataLoader(
        dataset = self.trainSet,
        batch_size = self.batch_size,
        shuffle = True
    )

    loss_func = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam( self.fc_net.parameters() )

    #train
    for epoch in tqdm.tqdm( range(self.epochs) ):
      #print( '\nepoch: ', epoch )
      for step, (batch_data, batch_label) in enumerate(trainLoader):
      
        batch_data = Variable( batch_data.to(device) )
        batch_label = Variable( batch_label.to(device) )

        y_predict = self.fc_net( batch_data )
        loss = loss_func( y_predict, batch_label )
        #print( '\nloss = ', loss.data.cpu().numpy() )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #save model
    torch.save( self.fc_net.state_dict(), save_path )
    print('model saved at ', save_path )

  def predict(self, image):

    #we don't need to check size because keypoints will be normalized eventually
    data = np.expand_dims( image, axis=0 )
    landmarks = self.kpClassifier.images_to_landmarks(data)
    keypoints = self.kpClassifier.landmarks_to_float_array(landmarks)
    keypoints = keypoints.reshape( (keypoints.shape[0], keypoints.shape[1]*keypoints.shape[2]) )
    with torch.no_grad():
      fc_input = torch.FloatTensor( keypoints )
      prediction = self.fc_net( fc_input )[0] #prediction: shape (29,)
      #print( 'prediction: ', torch.argmax(prediction, axis=0).cpu().numpy() ) 

    return prediction.cpu().numpy() #reduce one dimension

  def test(self, size=500):

    #check device
    if torch.cuda.is_available(): 
      device = 'cuda'
      self.fc_net.cuda()
    else: device = 'cpu'

    with torch.no_grad():

      test_data, test_label = readData( self.TEST_PATH, small=True, size=size )
      test_data = self.kpClassifier.images_to_landmarks(test_data)
      test_data = self.kpClassifier.landmarks_to_float_array(test_data)
      test_data = test_data.reshape( (test_data.shape[0], test_data.shape[1]*test_data.shape[2]) )

      test_label = torch.FloatTensor( test_label ).to(device)
      test_label_flat = torch.argmax( test_label, dim=-1 ).cpu().numpy()

      #eliminate the influence of not detecting a hand seccessfully
      for i in range(len(test_data)):
        if not test_data[i].any():
          test_label_flat[i] = 27 #change label to nothing

      #go through fc net
      test_data = torch.FloatTensor( test_data ).to(device)
      test_predict = self.fc_net( test_data )
      test_predict_flat = torch.argmax( test_predict, dim=-1 ).cpu().numpy()
      
      analyse_confusion_matrix( test_label.cpu().numpy(), test_predict.cpu().numpy() )

      print( '\ny_predict:\n ', test_predict_flat )
      print( '\ngt: \n', test_label_flat ) 
      accuracy = ( np.sum( test_predict_flat == test_label_flat ) )/len(test_predict_flat)
      print( 'accuracy = ', accuracy )
    
      precision, recall, f1, support = precision_recall_fscore_support( test_label_flat, test_predict_flat, average='macro', zero_division=1)
      precision = round(precision, 4)
      recall = round(recall, 4)
      f1 = round(f1, 4)
      
      print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


'''========== class ASL_alphabet_kp_nb_classifier =========='''
class ASL_alphabet_kp_nb_classifier:
  '''a class that make use of keypointsClassifier + NB classifier to perform training, testing and inference'''

  def __init__(self, TRAIN_PATH=None, TEST_PATH=None):
    self.kpClassifier = keypointsClassifier()
    self.TRAIN_PATH = TRAIN_PATH
    self.TEST_PATH = TEST_PATH 
    self.nb_model = None

  def load_nb_model(self, path):
    pass

  def train(self, size=2500):

    train_images, train_labels = readData( self.TRAIN_PATH, small=True, size=size )
    train_labels = np.array( [ np.argmax( one_hot ) for one_hot in train_labels ] )
    
    train_landmarks = self.kpClassifier.images_to_landmarks( train_images )
    train_keypoints = self.kpClassifier.landmarks_to_float_array( train_landmarks )

    #GaussianNB only accepts 2-dim data, so we flatten the keypoints
    train_keypoints = train_keypoints.reshape( (train_keypoints.shape[0], train_keypoints.shape[1]*train_keypoints.shape[2]) )

    nb_model = GaussianNB()
    nb_model.fit( train_keypoints, train_labels )

    #save model
    self.nb_model = nb_model

  def test(self, size=500):

    test_images, test_labels = readData( self.TEST_PATH, small=True, size=size )
    test_labels = np.array( [ np.argmax( one_hot ) for one_hot in test_labels ] )

    test_landmarks = self.kpClassifier.images_to_landmarks( test_images )
    test_keypoints = self.kpClassifier.landmarks_to_float_array( test_landmarks )

    test_keypoints = test_keypoints.reshape( (test_keypoints.shape[0], test_keypoints.shape[1]*test_keypoints.shape[2]) )
    y_predict = self.nb_model.predict(test_keypoints)

    analyse_confusion_matrix( test_labels, y_predict )

    print( '\ny_predict:\n ', y_predict )
    print( '\ngt: \n', test_labels ) 
    accuracy = ( np.sum( y_predict == test_labels ) )/len(y_predict)
    print( 'accuracy = ', accuracy )

    precision, recall, f1, support = precision_recall_fscore_support( test_labels, y_predict, average='macro', zero_division=1)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")
