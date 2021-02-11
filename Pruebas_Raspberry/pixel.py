import board
import neopixel
from time import sleep
from os import urandom
from datetime import datetime

POWER_GREEN = (60, 255, 51)
GREEN = (0,5,0)
POWER_RED = (237,41,10)
RED =       (5, 0, 0)
BLUE =      (0, 0, 5)
YELLOW =    (6, 5, 0)
PINK =      (3, 1, 1)
BABY_BLUE = (0, 1, 3)
ORANGE =    (6, 2, 0)
PURPLE =    (4, 0, 6)
GREY =      (57,23,18)

COLORS = [GREEN, RED, BLUE, YELLOW, PINK, BABY_BLUE, ORANGE, PURPLE]


letters = {
    'A': ([61,62,63,64,66,125,127,128,129,130,189,191,192,194], 3),
    'B': ([61,62,63,64,66,126,127,128,129,130,189,191,192,193,194], 3),
    'C': ([61,62,63,64,127,128,191,192,193,194], 3),
    'D': ([62,63,64,66,125,127,128,130,189,191,192,193], 3),
    'E': ([61,62,63,64,127,128,129,191,192,193,194], 3),
    'F': ([61,62,63,64,126,127,128,191,192], 3),
    'G': ([61,62,63,64,127,128,130,131,191,188,192,193,194,195], 4),
    'H': ([61,63,64,66,125,126,127,128,129,130,189,191,192,194], 3),
    'I': ([61,62,63,65,126,129,190,192,193,194], 3),
    'J': ([60,61,62,66,125,130,189,191,193,194], 4),
    'K': ([60,63,64,66,126,127,128,129,189,191,192,195], 4),
    'L': ([63,64,127,128,191,192,193,194], 3),
    'M': ([59,63,64,65,67,68,123,125,127,128,132,187,191,192,196], 5),
    'N': ([60,63,64,67,124,126,127,128,130,131,188,191,192,195], 4),
    'Ñ': ([61,62,124,127,128,129,131,188,189,191,192,195], 4),
    'O': ([61,62,63,64,66,125,127,128,130,189,191,192,193,194], 3),
    'P': ([61,62,63,64,66,125,127,128,129,130,191,192], 3),
    'Q': ([61,62,63,64,66,125,127,128,130,189,191,192,193,194,195], 4),
    'R': ([61,62,63,64,66,125,126,127,128,129,189,191,192,194], 3),
    'S': ([61,62,63,64,126,127,129,130,189,192,193,194], 3),
    'T': ([61,62,63,65,126,129,190,193], 3),
    'U': ([61,63,64,66,125,127,128,130,189,191,192,193,194], 3),
    'V': ([61,63,64,66,125,127,128,130,189,191,193], 3),
    'W': ([59,63,64,68,123,127,128,130,132,187,189,191,193,195], 5),
    'X': ([61,63,64,66,126,129,189,191,192,194], 3),
    'Y': ([61,63,64,66,125,127,129,190,193], 3),
    'Z': ([61,62,63,66,125,129,191,192,193,194], 3),
    'a': ([64,65,66,125,128,129,130,189,191,192,193,194], 3),
    'b': ([64,127,128,129,130,189,191,192,193,194], 3),
    'c': ([64,65,66,127,128,191,192,193,194], 3),
    'd': ([66,125,128,129,130,189,191,192,193,194], 3),
    'e': ([65,66,124,127,128,130,131,191,193,194], 4),
    'f': ([64,65,66,127,128,129,191,192], 3),
    'g': ([64,65,66,125,127,128,129,130,189,192,193,194], 3),
    'h': ([64,127,128,129,130,189,191,192,194], 3),
    'i': ([64,128,191,192], 1),
    'j': ([66,130,189,191,193,194], 3),
    'k': ([64,66,125,127,128,129,189,191,192,194], 3),
    'l': ([64,127,128,191,192,193], 2),
    'm': ([64,68,123,124,126,127,128,130,132,187,191,192,196], 5),
    'n': ([64,67,124,127,128,129,131,188,189,191,192,195], 4),
    'ñ': ([65,66,128,129,131,188,189,191,195], 4),
    'o': ([64,65,66,125,127,128,130,189,191,192,193,194], 3),
    'p': ([64,65,66,125,127,128,129,130,191,192], 3),
    'q': ([64,65,66,125,127,128,129,130,189,194], 3),
    'r': ([64,125,126,127,128,130,191,192], 3),
    's': ([64,65,66,127,128,129,130,189,192,193,194], 3),
    't': ([64,65,66,126,129,190,193], 3),
    'u': ([64,66,125,127,128,130,189,191,192,193,194], 3),
    'v': ([64,66,125,127,128,130,189,191,193], 3),
    'w': ([64,68,123,127,128,130,132,187,189,191,193,195], 5),
    'x': ([64,66,125,127,129,189,191,192,194], 3),
    'y': ([64,66,125,127,129,190,193], 3),
    'z': ([64,65,66,125,129,191,192,193,194], 3),
    ' ': ([], 1),
    '1': ([62,64,65,126,129,190,192,193,194], 3),
    '2': ([61,62,63,66,125,129,191,192,193,194], 3),
    '3': ([62,63,66,126,129,189,192,193], 3),
    '4': ([61,63,64,66,125,127,128,129,130,189,194], 3),
    '5': ([61,62,63,64,125,126,127,130,189,192,193,194], 3),
    '6': ([61,62,63,64,127,128,129,130,189,191,192,193,194], 3),
    '7': ([61,62,63,66,125,129,130,131,189,194], 3),
    '8': ([61,62,63,64,66,125,126,127,128,129,130,189,191,192,193,194], 3),
    '9': ([61,62,63,64,66,125,126,127,130,189,194], 3),
    '0': ([61,62,63,64,66,125,127,128,130,189,191,192,193,194], 3),
    '.': ([192], 1),
    ',': ([190,192], 2),
    ':': ([127,191,], 1),
    ';': ([126,190,192], 2),
    '!': ([63,64,127,128,192], 1),
    '?': ([62,64,66,125,129,193], 3),
    '¿': ([62,126,128,189,191,193], 3),
    '"': ([61,63,64,66], 3),
    "'": ([63,64], 1),
    "'": ([63,64], 1),
    '-': ([128,129,130], 3),
    '+': ([126,128,129,130,190], 3),
    '*': ([64], 1),
    '/': ([60,66,125,129,190,192], 4),
    '(': ([62,64,127,128,191,193], 2),
    ')': ([63,65,126,129,190,192], 2),
    '=': ([126,127,190,191], 2),
    '[': ([62,63,64,127,128,191,192,193], 2),
    ']': ([62,63,65,126,129,190,192,193], 2),
    '{': ([61,62,65,127,128,190,193,194], 3),
    '}': ([62,63,65,125,130,190,192,193], 3),
    '@': ([60,61,65,68,122,127,128,130,131,133,186,188,190,194,196],6),
    '$': ([61,62,63,64,126,127,129,130,189,192,193,194], 3),
    "^":([60,62,65,67,128,132,188,189,190],5),
    "ç":([60,62,65,67,129,130,131,187,191],5),
    "{":([60,62,65,67,128,132,188,189,190,194],5)
    }


def draw_letter(letter, pos, np, color):    
    for point in letter:
        for r in [range(32,64), range(64,96), range(96,128), range(128,160), range(160,192), range(192,224)]:
            if(r.stop % 64 == 0):
                if (point in r and point - pos in r):                
                    np[point - pos] = color
            
            else:
                if (point in r and point + pos in r):                
                    np[point + pos] = color
                
def clean(letter, pos, np):
    color = (0,0,0)
    for point in letter:
        for r in [range(32,64), range(64,96), range(96,128), range(128,160), range(160,192), range(192,224)]:
            if(r.stop % 64 == 0):
                if (point in r and point - pos in r):                
                    np[point - pos] = color
            
            else:
                if (point in r and point + pos in r):                
                    np[point + pos] = color
    np.write()

def scroll_text(text, np, speed, color):    
    text = list(text)
    screen = []  
    while text or screen:
        if screen:
            if (screen[0]['pos'] + screen[0]['data'][1]) <= 0:
                # Comienza a eliminarse el texto por el lado izquierdo
                screen.pop(0)
            if text and ((screen[-1]['pos'] + screen[-1]['data'][1]) < 32):                
                screen.append({'data': letters[text.pop(0)], 
                               'pos': 32, 
                               'color': color})
        else:  # Se preparan las letras para proyectar
            screen.append({'data': letters[text.pop(0)], 
                           'pos':32, 
                           'color': color})
        for letter in screen:  # Escribimos la letra en la pantalla
            draw_letter(letter['data'][0], letter['pos'], np, (170,255,100))
            letter['pos'] -= 1  # Desplazamiento
        np.write() 
        np.fill([0,0,0])
        sleep(.2/speed)

def accuracy_clean(np):    
    for i in range(255,223,-1):
        np[i] = (0,0,0)
    np.write() 

def accuracy_print(acc, np):
    accuracy_clean(np)
    pixels_num = int(acc*32)
    pixels_num = 32- pixels_num
    if(acc >0.7):
        color = (87,230, 0)
    else:
        if (acc > 0.4):
            color = (231,76,60)
        else:            
            color= (100,0,0)


    for i in range(255,223+pixels_num,-1):
        np[i] = color
    np.write() 




def write_text(text, np, color,border):    
    text = list(text)
    screen = [] 
    np.fill([0,0,0])
    np.write()
    empty = True
    
    while empty:
        np.fill([0,0,0])
        if screen:
            if (screen[0]['pos'] <= border):
                # Las lestras salen de la pantalla
                empty = False
            if text and ((screen[-1]['pos'] + screen[-1]['data'][1]) < 32):               
                screen.append({'data': letters[text.pop(0)], 
                               'pos': 32, 
                               'color': color})
        else:  # Se rellena la pantalla a mostrar
            screen.append({'data': letters[text.pop(0)], 
                           'pos':32, 
                           'color': color})
        for letter in screen:  # Imprimir pantalla
            draw_letter(letter['data'][0], letter['pos'], np, color)
            letter['pos'] -= 1 # Desplazamiento
 
    np.write()         

def digits(a):    
    c = int(a%10)
    a = int(a/10)
    b = int(a%10)
    return b,c

def zero(np,color):
    draw_letter(letters[str(0)][0], 4, np, color)
    draw_letter(letters[str(0)][0], 8, np, color)
    draw_letter(letters[str(0)][0], 14, np, color)
    draw_letter(letters[str(0)][0], 18, np, color)
    draw_letter(letters[str(0)][0], 24, np, color)
    draw_letter(letters[str(0)][0], 28, np, color)



def clock(np,color):       
    np.fill([0,0,0]) 
    h1, h2, m3, m4, s5, s6 = [0,0,0,0,0,0]
    draw_letter(letters[':'][0], 12, np, color)    
    draw_letter(letters[':'][0], 22, np, color) 
    zero(np,color)
    while True:   
        now = datetime.now()            
        d = now.hour    
        d1, d2 = digits(d) 

        d = now.minute
        d3, d4 = digits(d)

        d = now.second
        d5,d6 = digits(d)

        if h1!= d1:
            clean(letters[str(h1)][0], 4, np)
            h1=d1
            draw_letter(letters[str(h1)][0], 4, np, color)
            np.write()
        if h2!= d2:
            clean(letters[str(h2)][0], 8, np)
            h2=d2                 
            draw_letter(letters[str(h2)][0], 8, np, color)
            np.write()

        if m3!= d3:
            clean(letters[str(m3)][0], 14, np)
            m3=d3             
            draw_letter(letters[str(m3)][0], 14, np, color)
            np.write()

        if m4!= d4:
            clean(letters[str(m4)][0], 18, np)
            m4=d4                  
            draw_letter(letters[str(m4)][0], 18, np, color)
            np.write()

        if s5!= d5:
            clean(letters[str(s5)][0], 24, np)
            s5=d5                  
            draw_letter(letters[str(s5)][0], 24, np, color)
            np.write()

        if s6!= d6:
            clean(letters[str(s6)][0], 28, np)
            s6=d6                  
            draw_letter(letters[str(s6)][0], 28, np, color)
            np.write() 
           
  

