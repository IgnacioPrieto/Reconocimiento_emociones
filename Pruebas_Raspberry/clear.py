import os
import pixel 
import board
import neopixel
import time
import _thread

panel = neopixel.NeoPixel(board.D18, 256,auto_write=False)
panel.fill([0,0,0])
panel.write()