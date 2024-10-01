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

# Iterar sobre los folds
for fold_index, (train_index, val_index) in enumerate(kf.split(images), 1):
    # Construir el modelo de CNN
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        #Conv2D(128, (3, 3), activation='relu'),
        #MaxPooling2D(2, 2),
        
       # Conv2D(256, (3, 3), activation='relu'),
        #MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(32, activation='relu'),# kernel_regularizer=l1(0.0001)),
        
        #Dense(32, activation='relu'),# kernel_regularizer=l1(0.0001)),
        
        #Dense(16, activation='relu'),# kernel_regularizer=l1(0.0001)),
        
        #Dense(8, activation='relu'),# kernel_regularizer=l1(0.0001)),
        
        #Dense(4, activation='relu'),# kernel_regularizer=l1(0.0001)),
        
        Dense(1, activation='sigmoid')
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    X_train, X_val = images[train_index], images[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    
    # Entrenar el modelo en este fold
    history = model.fit(X_train, y_train, epochs=400, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Guardar el modelo entrenado
    model.save(f"C:/Users/Usuario/Desktop/temporal/3-64_1-256_{fold_index}.h5")
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Número de épocas
    epochs2 = range(1, len(loss) + 1)

    # Grafica la pérdida y el accuracy de entrenamiento
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs2, loss, 'b', label='Pérdida de entrenamiento')
    plt.plot(epochs2, val_loss, 'r', label='Pérdida de validación')
    plt.title('Pérdida de entrenamiento y validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs2, accuracy, 'b', label='Accuracy de entrenamiento')
    plt.plot(epochs2, val_accuracy, 'r', label='Accuracy de validación')
    plt.title('Accuracy de entrenamiento y validación')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"C:/Users/Usuario/Desktop/temporal/3-64_1-256_{fold_index}.png")


#Codigo completo original"