#pipline.py
import numpy as np
from spellchecker import SpellChecker
from collections import deque
from .dataset import int_to_string

class charToWordProcessor:
    '''
    1. Take sequence of integer between 0 ~ 28 as input
    2. output word whenever a 'space' is taken
    3. check spelling as well ( output word must be in the dictionary )
    '''
    def __init__(self):
        self.buf = deque()
        self.checker = SpellChecker()
        
    def process(self, pred):
        '''
        pred: is an integer between 0 to 28
        '''
        
        if pred == 26: #'del'
            '''delete last char'''
            if self.buf: #if buf is not empty
                self.buf.pop()

        elif pred == 27: #'nothong'
            pass #do nothing
        
        elif pred == 28: #'space'
            '''output a word'''
            original = ''
            while self.buf: #while buf is not empty
                original += self.buf.popleft()
            corrected = self.checker.correction( original )
            print( corrected )
            
        else: #'A'~'Z'
            '''append a char'''
            input_char = chr( ord('a') + int(pred) )
            self.buf.append( input_char )

class pipeline:
    '''
    1. key frame -> classification result
    2. sequence of classification results -> words
    '''
    def __init__(self, classifier ):
     
        self.processor = charToWordProcessor()
        self.image_classifier = classifier
    
    def one_step(self, frame):
        
        pred = self.image_classifier.predict( frame ) #return an numpy array of shape (29,)
        index = np.argmax(pred, axis = 0)
        if index != 27: print('Predicted label: ', int_to_string(index) )
        self.processor.process( index )
      