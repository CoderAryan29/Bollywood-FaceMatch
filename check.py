import tensorflow as tf
import keras
import numpy as np
from mtcnn import MTCNN

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("NumPy:", np.__version__)
print("MTCNN imported successfully!")

from keras_vggface.vggface import VGGFace

print("VGGFace imported successfully!")