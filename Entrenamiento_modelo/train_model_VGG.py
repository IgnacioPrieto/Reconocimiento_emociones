# LIBRERÍAS
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  #Evita complicaciones con CUDA (Si no lo utilizas, coméntala)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' Descomenta esta línea para forzar el uso de CPU. Si no usas CUDA, siempre se ejecutará por CPU por lo que no es necesaria
import sys
import subprocess
from subprocess import call
import numpy as np
import pandas as pd
import tensorflow as tf
import keyboard
import traceback
import datetime
from os import remove
import tensorflow.compat.v1.keras.utils as np_utils
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn import metrics, decomposition
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix



# PARÁMETROS
PRE_TRAINED_MODEL = "VGG16"
IMG_SIZE = 48
EPOCH = 10000
NUM_TRAIN = 28708
NUM_VALIDATION = 7178
NUM_CLASSES = 7
BATCH = 64 
LEARNING_RATE = 0.0005
DECAY = 1e-5
PATIENCE = 50

# DIRECCTORIOS
CSV_FILE= 'dataset.csv'
TIME = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
SESSION_PATH= "run_" + TIME +"\\"
LOG_CSV = SESSION_PATH + "log_csv" + ".log"
LOGS = "logs\\"
FIT = "logs\\"+ "fit\\"
LOG_DIR =  FIT+ TIME
CKPT_DIR = SESSION_PATH+ "ckpt\\"

if not os.path.exists(SESSION_PATH):
    os.mkdir(SESSION_PATH)
    
if not os.path.exists(CKPT_DIR):
    os.mkdir(CKPT_DIR)    

if not os.path.exists(LOGS):
    os.mkdir(LOGS)

if not os.path.exists(FIT):
    os.mkdir(FIT)

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)  
            

#VERSIONES
print("TensorFlow version: ", tf.__version__)
print("Versión de Python: ",sys.version)

# CLASE
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}):
        if(epoch % 5 == 0) :                            
            files_to_delete = os.listdir(SESSION_PATH + "ckpt\\")
            # Ordenamos la lista de ficheros alfabeticamente
            files_to_delete.sort(reverse=True)   
            # Solo puede haber dos ficheros, csv.log y el mejor checkpoint 
            while(len(files_to_delete)>1):
                # Se borra el archivo  
                try: 
                     remove(SESSION_PATH + "ckpt\\" + files_to_delete[1])
                     del(files_to_delete[1])
                except Exception:
                    print("Un archivo aún no ha podido ser eliminado, se eliminará más tarde")
        
        if keyboard.is_pressed('ç'):
            print('Se aborto manualmente el entrenamiento')
            self.model.stop_training = True 
        
        
                
#FUNCIONES
def load_data(csv_dir):
    ### CARGAR DATOS DEL CSV
    # Todo los datos
    print('Comienza la importación de los datos')
    # Análisis de los datos
    data_csv = pd.read_csv(csv_dir, header=0)
    print(data_csv.Usage.value_counts())     
    print('La base de datos contiene {} caras'.format(data_csv.shape[0]))
    emotion_list = list(data_csv.emotion)
    pixels_list = list(data_csv.pixels)

    # Se transforman las emociones a un array categorical
    emotion_array_np = np_utils.to_categorical(np.asarray(emotion_list), num_classes=NUM_CLASSES, dtype='float32')
    images_array_np = create_images_array(pixels_list)
    return emotion_array_np, images_array_np
    

def create_images_array(pixels_list):
        # Construimos un vector de matrices, donde cada matriz será una foto
        images_array = []
        for image in pixels_list:
        
                image_matrix = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                pixels_array = image.split()
                
                for row_num in range(IMG_SIZE):
                        start_row = row_num * IMG_SIZE  
                        finish_row = start_row + IMG_SIZE    
                        image_matrix[row_num] = pixels_array[ start_row : finish_row] 
                                               
                images_array.append(np.array(image_matrix))

        images_array_np = (np.array(images_array)).astype('float32') / 255.0  
        return images_array_np


def apply_PreTrained_model(model,data):
    #Se crea el modelo pre entrenado selecionado 
    #include_top=False no incluye las ultimas capas softmax
    if model== "VGG16":
        CNN_model = VGG16(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights='imagenet')  
           
    if model== "VGG19":
        CNN_model = VGG19(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights='imagenet')
            
    result = CNN_model.predict(data)       
    #Se aplica a los datos y se retorna el modelo para contruir el modelo final completo      
    return result, CNN_model

    
def main(): 
    # Obtenemos dos numpy array con las emociones y las imagenes en orden
    emotion_array_np, images_array_np  = load_data(CSV_FILE)
    
    # Triplicamos las imágenes a 3 capas, para tenerlas en formato RGB, shape(NUM,48,48) -->  shape(NUM,48,48,3)
    images_array_np = np.broadcast_to(images_array_np[...,None],images_array_np.shape+(3,))   
    images_original_np = images_array_np

    images_train_original = images_original_np[0:NUM_TRAIN]
    images_test_original = images_original_np[NUM_TRAIN+1:]

    # Guardamos el formato de entrada al modelo
    initial_inputs = Input(np.shape(images_array_np[0]))
    images_array_np, PreTrained_Model = apply_PreTrained_model(PRE_TRAINED_MODEL,images_array_np)

    # FORMATO ENTRADA shape(NUM,48,48,3)  
    # SE DIVIDE EN:
    # TRAIN
    emotion_train = emotion_array_np[0:NUM_TRAIN]
    images_train = images_array_np[0:NUM_TRAIN]

    # TEST    
    emotion_test = emotion_array_np[NUM_TRAIN+1:]
    images_test = images_array_np[NUM_TRAIN+1:]
    
    # Eliminamos el csv inicial
    del (images_array_np, emotion_array_np)    
    

   #Construimos la red neuronal 
    neuronal_network = Sequential()    
            
    #Completamos el modelo con nuestras capas
    # Capa input
    neuronal_network.add(Dense(256, input_shape= np.shape(images_train[0]) , activation='relu'))
    neuronal_network.add(Dropout(0.25))
    # Capa 1
    neuronal_network.add(Dense(128, input_shape= (256,) , activation='relu'))
    neuronal_network.add(Dropout(0.25))
    # Capa 2
    neuronal_network.add(Dense(64, input_shape=(128,)))
    neuronal_network.add(Dropout(0.25))
    # Capa Out
    neuronal_network.add(Dense(NUM_CLASSES, activation='softmax'))
     
    neuronal_network.compile(loss='categorical_crossentropy',
                    optimizer=Adamax(lr= LEARNING_RATE, decay = DECAY),metrics=['accuracy'])
      
            
    neuronal_network.compile(loss='categorical_crossentropy', optimizer=Adamax(lr= LEARNING_RATE, decay= DECAY),metrics=['accuracy'])
    
    # Gráficos a timepo real
    # Se define tensorboard para seguir el entrenamiento a tiempo real   
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

    try:     
        # Lanzamos los tensorboard 
        process = subprocess.Popen('tensorboard --logdir=./logs', stdout=subprocess.PIPE, stderr=subprocess.PIPE)            
        # Abrimos el navegador 
        os.system('start chrome http://localhost:6006/')
    except Exception:
        print ("No pudo lanzarse tensorboard. Excepción: \n", traceback.print_exc())  
    

    # CALLBACKS
    checkpoint_name = 'val_acc_{val_accuracy:.4f}-{epoch:02d}.hdf5'
    checkpoint_filepath = CKPT_DIR + checkpoint_name
    csv_logger = tf.keras.callbacks.CSVLogger(LOG_CSV, append=False)
    # early_stop = tf.keras.callbacks.EarlyStopping('val_loss', patience=PATIENCE)      
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', verbose=1, mode='max', save_best_only=True) 
    callbacks = [myCallback(), tensorboard, model_checkpoint, csv_logger]
    
    # ENTRENAMIENTO
    model_info = neuronal_network.fit(
            images_train,emotion_train,
            batch_size= BATCH,        
            epochs=EPOCH,
            validation_data= (images_test,emotion_test),
            verbose=1,
            callbacks= callbacks
            )
    
    #CONSTRUCCIÓN DEL MODELO COMPLETO  
    neuronal_input = PreTrained_Model(initial_inputs)    
    model_output = neuronal_network(neuronal_input)
    
    #Pre trained model + neuronal_network 
    convolutional_model = Model( initial_inputs, model_output)    
    convolutional_model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy']) 
    
    # GUARDADO DE LA ESTRUCTURA Y CONFIGURACIÓN EN UN .txt
    file = open(SESSION_PATH + "info_model.txt", "w")    
    convolutional_model.summary(print_fn=lambda x: file.write(x + '\n'))    
    file.write("\n\n" + "Learning rate: " + str(LEARNING_RATE) + "\nDecay: " + str(DECAY))    
    file.close()

    #Evaluamos el modelo final 
    results = neuronal_network.evaluate(images_test, emotion_test, batch_size=128)
    
    # GUARDADO DEL MODELO 
    accuracy = model_info.history['accuracy']   
    epochs = str(len(accuracy))
    a = round(results[1], 3)
    accuracy_value = str(a)

    model_name = "model_e_"+ epochs + "-acc_" + accuracy_value +".hdf5"
    try:          
        convolutional_model.save(SESSION_PATH + model_name )
    except Exception:
        print("Error al guardar el modelo \n", traceback.print_exc())        
    
      
if __name__ == "__main__":  
    main()

    
    