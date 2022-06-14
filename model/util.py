#util.py
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def analyse_confusion_matrix( y_true, y_pred ):
  '''
  y_true, y_pred are of shape (n_samples, 29) 
  y_pred is the probability for each class
  '''
  y_true = np.argmax( y_true, axis=1 )
  y_pred = np.argmax( y_pred, axis=1 )
  confusion_mat = confusion_matrix( y_true, y_pred, normalize='true' )
  dis = ConfusionMatrixDisplay( confusion_mat )
  dis.plot()

  
  accuracy = np.sum( y_true == y_pred )/len(y_true)
  print('accuracy = ', accuracy )
