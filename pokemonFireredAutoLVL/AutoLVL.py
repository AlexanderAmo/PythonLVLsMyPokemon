import numpy as np
from PIL import ImageGrab
import cv2
import time
from Controller import PressKey,ReleaseKey, W, A, S, D, Z, X, E
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#Z = A, X = B, E = start

def reset(numberOfSearches):
    if(numberOfSearches <= 1):
        PressKey(W)
        time.sleep(.025)
        ReleaseKey(W)
        time.sleep(.1)    
    PressKey(S)
    time.sleep(.025)
    ReleaseKey(S)
    time.sleep(.1)
    PressKey(A)
    time.sleep(.025)
    ReleaseKey(A)
    time.sleep(.1)
    PressKey(W)
    time.sleep(.025)
    ReleaseKey(W)
    time.sleep(.1)
    PressKey(S)
    time.sleep(.025)
    ReleaseKey(S)
    time.sleep(1)
    
def spamButton(Button, durationInSeconds):
    timeEnd = time.time() + durationInSeconds
    while (time.time() <  timeEnd):
        PressKey(Button)
        time.sleep(.03)
        ReleaseKey(Button)
        time.sleep(.1)

def dontLearnMoves():
    PressKey(Z)
    time.sleep(.01)
    ReleaseKey(Z)
    time.sleep(.05)
    PressKey(S)
    time.sleep(.01)
    ReleaseKey(S)
    time.sleep(.05)
    PressKey(Z)
    time.sleep(.01)
    ReleaseKey(Z)
    time.sleep(.05)
    PressKey(Z)
    time.sleep(.01)
    ReleaseKey(Z)
    time.sleep(.05)
    PressKey(Z)
    time.sleep(.01)
    ReleaseKey(Z)
    time.sleep(.05)
    
def runAway():
    PressKey(Z)
    time.sleep(.03)
    ReleaseKey(Z)
    time.sleep(.1)
    PressKey(S)
    time.sleep(.03)
    ReleaseKey(S)
    time.sleep(.1)
    PressKey(D)
    time.sleep(.03)
    ReleaseKey(D)
    time.sleep(.1)
    PressKey(Z)
    time.sleep(.03)
    ReleaseKey(Z)
    time.sleep(.1)
    PressKey(Z)
    time.sleep(.03)
    ReleaseKey(Z)
    time.sleep(.1)

def heal():
    print('healing')
    PressKey(W)
    time.sleep(1)
    ReleaseKey(W)
    time.sleep(.1)
    PressKey(A)
    time.sleep(.24)
    ReleaseKey(A)
    time.sleep(.1)
    PressKey(W)
    time.sleep(1.8)
    ReleaseKey(W)
    time.sleep(.1)
    spamButton(Z, 3.28)
    spamButton(X, 1.5)
    time.sleep(.1)
    PressKey(S)
    time.sleep(2.2)
    ReleaseKey(S)
    PressKey(D)
    time.sleep(.205)
    ReleaseKey(D)
    PressKey(S)
    time.sleep(.65)
    ReleaseKey(S)
    
def findWildPokemon():
    print('Finding Pokemon')
    time.sleep(.1)
    PressKey(D)
    time.sleep(.025)
    ReleaseKey(D)
    time.sleep(.1)
    PressKey(W)
    time.sleep(.025)
    ReleaseKey(W)
    time.sleep(.1)
    PressKey(A)
    time.sleep(.025)
    ReleaseKey(A)
    time.sleep(.1)


def get_text(img):
    text = pytesseract.image_to_string(img)
    return text
    
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(original_img):
    processed_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
    vertices = np.array([[10,500], [10,300],[800,300], [800,500]])
    processed_img = roi(processed_img, [vertices])
  
    return processed_img
  
def main():
    numberOfSearches = 0
    PP = 10
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0, 55, 700, 530)))
        new_screen = process_img(screen)
        text = get_text(new_screen)

        #cv2.imshow('window',np.array(screen))
        
        #auto level part of script
        if(PP > 0):
            if('Wild' in text and ('apeeared' in text or 'appeared' in text)):
                #print(text)
                print('commiting genecide')
                spamButton(Z, 4.3)
                if('learn' in text and ('apeeared' in text or 'appeared' in text)):
                    print(text)
                    dontLearnMoves()
                PP = PP - 1
                print('pp = ')
                print(PP)
                
                if(PP == 0):
                    numberOfSearches = 0
                    time.sleep(1)
                    PressKey(W)
                    time.sleep(.06)
                    ReleaseKey(W)
                    time.sleep(1)
                    if('Wild' in text and ('apeeared' in text or 'appeared' in text)):
                        print('running away')
                        runAway()
                    heal()
                    PP = 10
                else:
                    print('reset')
                    reset(numberOfSearches)
                    numberOfSearches = 0
            else:
                findWildPokemon()
                numberOfSearches += 1
                print('number of searches = ', numberOfSearches)
        last_time=time.time()
        #cv2.imshow('window', new_screen)   
        if cv2.waitKey(25) & 0xFF == ord('p'):
            cv2.destroyAllWindows()
            break
 
for i in list(range(4))[::--1]:
    print(i+1)
    time.sleep(1)

main()

