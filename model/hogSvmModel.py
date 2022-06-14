#HogSvmModel.py
import cv2
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

class ASL_alphabet_hog_svm_classifier():
    
    def __init__(self, TRAIN_PATH = None, TEST_PATH = None  ):
        
        self.TRAIN_PATH = TRAIN_PATH 
        self.TEST_PATH = TEST_PATH
        
        self.featureNum = 3780
        self.winSize = (64,128) 
        self.blockSize = (16,16)
        self.blockStride = (8,8)
        self.cellSize = (8,8) 
        self.Bin = 9
        self.hog = cv2.HOGDescriptor(self.winSize,self.blockSize,self.blockStride,self.cellSize,self.Bin)

        
    def load_model(self, model_path):
        
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)
        
    def predict(self, image):
        
        img = cv2.resize(image, (64, 128), cv2.COLOR_BGR2GRAY)
        hist = self.hog.compute(img,(8,8))
        feature = np.zeros((1, self.featureNum),np.float32)
        feature[0] = hist.reshape(-1)
        return self.clf.predict_proba(feature).reshape(-1)
    
    def train(self ,save_path='./hog_svm_model/hog_svm_model_unnamed.xml'):
        labelNum = 0
        i = 0
        for token in os.listdir(self.TRAIN_PATH):
            for image in os.listdir(self.TRAIN_PATH+"/"+token):
                labelNum += 1

        featureArray = np.zeros((labelNum, self.featureNum),np.float32)
        labelArray = np.zeros((labelNum, 1),np.int32)
        for token in os.listdir(self.TRAIN_PATH):
            if token == 'del':
                label = 26
            elif token == 'nothing':
                label = 27
            elif token == 'space':
                label = 28
            else:
                label = ord(token) - ord('A')

            for image in os.listdir(self.TRAIN_PATH+"/"+token):
                img=cv2.imread(self.TRAIN_PATH+"/"+token+"/"+image)
                img=cv2.resize(img, (64, 128), cv2.COLOR_BGR2GRAY)
                hist = self.hog.compute(img,(8,8))
                featureArray[i] = hist.reshape(-1)
                labelArray[i] = label
                i += 1
        self.clf = svm.SVC() 
        self.clf.probability=True
        self.clf.fit(featureArray,labelArray)
        with open(save_path, 'wb') as f:
            pickle.dump(self.clf, f)
            
    def test(self ):

        labelNum = 0
        i = 0
        for token in os.listdir(self.TEST_PATH):
            for image in os.listdir(self.TEST_PATH+"/"+token):
                labelNum += 1
        labelArray = np.zeros((labelNum, 1),np.int32).reshape(-1)
        predictArray = np.zeros((labelNum, 1),np.int32).reshape(-1)
        for token in os.listdir(self.TEST_PATH):
            if token == 'del':
                label = 26
            elif token == 'nothing':
                label = 27
            elif token == 'space':
                label = 28
            else:
                label = ord(token) - ord('A')

            for image in os.listdir(self.TEST_PATH+"/"+token):
                labelArray[i] = label
                img=cv2.imread(self.TEST_PATH+"/"+token+"/"+image)
                img=cv2.resize(img, (64, 128), cv2.COLOR_BGR2GRAY)
                hist = self.hog.compute(img,(8,8))
                feature = np.zeros((1, self.featureNum),np.float32)
                feature[0] = hist.reshape(-1)
                predictArray[i] = self.clf.predict(feature)
                i += 1
        labelArray = labelArray.reshape(-1)
        predictArray = predictArray.reshape(-1)
        precision, recall, f1, support = precision_recall_fscore_support( labelArray, predictArray, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
      
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


    

