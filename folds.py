import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping



    

