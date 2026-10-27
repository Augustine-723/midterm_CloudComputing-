import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import ExponentialDecay


root_path = os.path.abspath(os.path.dirname(__file__))

#hyperparameter
'''
======================================================================================
setting hyperparameter 
example:
learning rate = ...
epochs = ...
batch_size = ...
======================================================================================
'''


def main():
  #tensorboard logger setting
  log_dir = "logs/YOURMODEL"
  tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

  #Creating 25*...*5 neural network
  #last layer is doing softmax to five results
  '''
  ======================================================================================

  Setting your model here
  example:
  tf.keras.Sequential()
  keras.optimizers
  compile()
  others...
  Input:
       train_dataset = np.load(root_path + '/dataset/train.npz')
       train_data = train_dataset['data']
       train_label = 
  Output using one-hot-encoding
       to_categorical()


  ======================================================================================
  '''

  #Loading training dataset  
  '''
  ==code==
  '''
  #processing label
  '''
  ==code==
  '''
  #Training model
  model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])

  #Saving model
  model.save('YOURMODEL.h5')

  #Valid model
  #Loading Valid dataset
  '''
  ==code==
  '''
  #processing label with one-hot-encoding
  '''
  ==code==
  '''
  #prediction
  '''
  ====================================================================================== 
  example:
  predict()
  processing the prediction
  ======================================================================================
  '''
  # Calculating accuracy
  '''
  ==code==
  '''
  print(predicted_labels)
  print(true_labels)
  accuracy = np.mean(true_labels == predicted_labels)
  print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
  main() 

