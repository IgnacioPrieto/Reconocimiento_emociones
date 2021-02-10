import pixel
import board
import neopixel
import keyboard

# Conexión panel leds
panel = neopixel.NeoPixel(board.D18, 256, auto_write=False)
color = pixel.GREY
pixel.write_text("Cargando", panel, color,0)

# Tarda unos segundos, por ello se proyecta antes un mensaje
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os
from time import time
import _thread

### CONFIGURACIÓN ###
# Umbral de confianza de las respuestas (0-0.99)
ACCURACY = 0.10
# Se muestran las precisiones (True) de lo contrario (False)
ACC_ACTIVATED = False


# Activamos un hilo para no detener la ejecución
_thread.start_new_thread(pixel.scroll_text, ("Bienvenido", panel, 10, color))

# Archivo de modelo a utilizar
model='./modelos/modelo_CNN.hdf5'
CNN_model = tf.keras.models.load_model(model)


# Diccionario con las emociones 
emotion_dict = {0: "Enfadado", 1: "Asco", 2: "Asustado", 3: "Feliz", 4: "Triste", 5: "Sorpresa", 6: "Neutral"}
color_dict = {0: (255, 0, 0), 1: (129, 3, 203), 2: (255, 20, 20), 3: (87,255, 0), 4: (69, 59, 244), 5: (244, 59, 184), 6: (59, 244, 199)}
emotion = ""
state = True

cap = cv2.VideoCapture(0)   
while(state):       
    ret, frame = cap.read()

    # Se notifica el comienzo de la lectura hasta captar el primer rostro
    if(emotion == ""):
        pixel.write_text('leyendo', panel, color, 0)

    # Si no hay imagen, se termina el bucle
    if not ret:
        pixel.write_text('Error', panel, color, 0)        
        break
    haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.3, 5)
 
    if keyboard.is_pressed('ç'):
        state=False
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]      
        # Adaptacion dimensión           
        image = cv2.resize(roi_gray, (48, 48))
        np_image_array = image.astype('float32') / 255.0      
        img_ready = np.expand_dims(np.expand_dims(np_image_array, -1), 0) 

        # Inferencia       
        result = CNN_model.predict(img_ready)        
        emotion_number = int(np.argmax(result))

        # Actualización panel LED
        # Se comprueba si la probabilidad es superior al filtro marcado inicialmente
        if(np.max(result) > ACCURACY):
            # Se comprueba si la emoción ha cambiado            
            if(emotion_dict[emotion_number] != emotion):
                # Actualización de la emoción
                pixel.write_text(emotion_dict[emotion_number], panel, color_dict[emotion_number], 0)            
                emotion = emotion_dict[emotion_number]         
        
        # Valor demasiado bajo, no puede devolverse el resultado, por tanto ¿?
        else:            
            # Se comprueba que no se haya repetido ¿?
            if(emotion_dict[emotion_number] != emotion):
                # Actualización del panel                
                pixel.write_text("¿?", panel, (255,120,84), 0)
                emotion = "¿?"
                pixel.accuracy_clean(panel)
        # Se proyecta la precisión de la respuesta, si se indicó al inicio
        if(ACC_ACTIVATED):
            pixel.accuracy_print(np.max(result), panel)

cap.release()
pixel.scroll_text("Desconectando... hasta la proxima :)", panel, 10, color)

