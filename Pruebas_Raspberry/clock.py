import pixel
import board
import neopixel
import cv2
import keyboard
panel = neopixel.NeoPixel(board.D18, 256, auto_write=False)
color = (33, 130, 114)

#pixel.clock(panel, color)
emotion_dict = {0: "Enfadado", 1: "    Asco", 2: "Asustado", 3: "    Feliz", 4: "  Triste", 5: "Sorpresa", 6: " Neutral"}
color_dict = {0: (255, 0, 0), 1: (129, 3, 203), 2: (255, 20, 20), 3: (87,255, 0), 4: (69, 59, 244), 5: (244, 59, 184), 6: (59, 244, 199)}
pixel.write_text(emotion_dict[5], panel, color_dict[5], 0)    

cap = cv2.VideoCapture(0)   
while(True):
    ret, frame = cap.read()
    if keyboard.is_pressed('ç'):
        state=False 