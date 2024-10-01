import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
#from keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping

# Definir la ruta de la carpeta que contiene las imágenes
data_dir = 'C:/Users/Usuario/Desktop/Caninos'
classes = ['Canino', 'Central']  # Nombre de las subcarpetas/clases

# Definir el tamaño de las imágenes
img_width, img_height = 100, 75  # Ajusta según el tamaño de tus imágenes

early_stopping = EarlyStopping(monitor='val_accuracy', patience=60, restore_best_weights=True)

# Inicializar listas para almacenar las imágenes y sus etiquetas
images = []
labels = []

# Recorrer las carpetas/clases
for i, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = load_img(image_path, target_size=(img_width, img_height))
        image = img_to_array(image) / 255.0  # Normalizar los píxeles al rango [0, 1]
        images.append(image)
        labels.append(i)  # Asignar la etiqueta de la clase

# Convertir las listas a matrices NumPy
images = np.array(images)
labels = np.array(labels)

# Imprimir la forma de los datos cargados
print("Forma de las imágenes:", images.shape)
print("Forma de las etiquetas:", labels.shape)

# Inicializar KFold para la validación cruzada con 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)


    

