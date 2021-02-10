import keyboard
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os
import time
import _thread

### CONFIGURACIÓN ###
# Umbral de confianza de las respuestas (0-0.99)
ACCURACY = 0.10
# Se muestran las precisiones (True) de lo contrario (False) // Multi: mostrar una o varias caras 
ACC_ACTIVATED = False
MULTI = True
# Archivo de modelo a utilizar
model='./modelos/modelo_CNN.hdf5'
CNN_model = tf.keras.models.load_model(model)

# Diccionario con las emociones 
emotion_dict = {0: "Enfadado", 1: "Asco", 2: "Asustado", 3: "Feliz", 4: "Triste", 5: "Sorpresa", 6: "Neutral"}
color_dict = {0: (96, 100, 216), 1: (203, 3, 129), 2: (20, 20, 255), 3: (0,255, 87), 4: (244, 59, 69), 5: (184, 59, 244), 6: (199, 244, 59)}
emotion_string = ""
acc=""
state = True

# FPS Variables
prev_frame_time = 0
new_frame_time = 0
  
# Se activa la cámara
cap = cv2.VideoCapture(0)

while(state):
    ret, frame = cap.read()
    
    # Si no hay imagen, se termina el bucle
    if not ret:
        print('Error, no hay imagen')
        time.sleep(2)
        break

    haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:  
        new_frame_time = time.time()        
        roi_gray = gray[y:y + h, x:x + w]
        
        # Adaptacion dimensión           
        image = cv2.resize(roi_gray, (48, 48))
        np_image_array = image.astype('float32') / 255.0      
        img_ready = np.expand_dims(np.expand_dims(np_image_array, -1), 0) 

        # Inferencia       
        result = CNN_model.predict(img_ready)        
        emotion_number = int(np.argmax(result))

        if(np.max(result) > ACCURACY):
            # Actualización panel led
            if(emotion_dict[emotion_number] != emotion_string):       
                emotion_string = emotion_dict[emotion_number] 
        else:
            if(emotion_dict[emotion_number] != emotion_string):
                emotion_string="¿?"
                emotion_string = "¿?"
            
        #Cálculo de FPS
        fps = 1/(new_frame_time-prev_frame_time)  
        prev_frame_time = new_frame_time        
        fps = int(fps)
        color = color_dict[emotion_number] 
        

        #Cálculo de resultados
        if(ACC_ACTIVATED):
            acc = "  "+str(int(np.max(result)*100))+"%"
        if(MULTI):            
            cv2.putText(frame, emotion_string+acc , (x+30,y-30), cv2.FONT_HERSHEY_SIMPLEX,  1, color, 2)  
            cv2.putText(frame, "FPS: "+str(fps) , (230,450), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,0), 2)  
        else:
            cv2.rectangle(frame,(225,460),(590,420), (0,0,0),-1)
            cv2.putText(frame, emotion_string+acc+"FPS: "+str(fps) , (230,450), cv2.FONT_HERSHEY_SIMPLEX,  1, color, 2)                 
        
        
    cv2.imshow('Video', cv2.resize(frame,(1080,720)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                        
cap.release()
cv2.destroyAllWindows()
print("Desconectando... hasta la proxima :)")

