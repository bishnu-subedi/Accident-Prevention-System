"""Module for camera; captures image"""

import numpy as np
from picamera import PiCamera
import cv2
#cap = cv2.VideoCapture(0)

"""Camera setup...."""

camera = PiCamera()
camera.rotation = 180
camera.resolution = (80,40)    #1920 700



"""Takes pic from picam to create matrix for feeding of Network"""
def captureImage():
    """
    camera.start_preview()
    #sleep(1)
    camera.capture('/home/pi/Desktop/proj/image61.png')
    camera.stop_preview()

    #Extracting the picture
    img_file = '/home/pi/Desktop/proj/image61.png'
    gray_img = cv2.imread(img_file, flags=0)  # grayscale

    gray_img = cv2.resize(gray_img, (80, 40))
    



    #print('Gray shape:', gray_img.shape)
    #Turning it into gray-scale

    #plt.imshow(gray_img, plt.cm.binary)
    #plt.show()
    
    

    #Turning 2D into 1D
    pre_data = np.asarray(gray_img)
    pre_data = pre_data.flatten()
    #print(pre_data)



    
    #pre_data = pre_data/255
    #pre_data = pre_data.round(3)
    
    qwer = pre_data
    qwer[qwer>=165] = 230
    qwer[qwer<165] = 30

    
    

    #Deleting the picture on file stored
    call([command], shell=True)

    #return pre_data
    return qwer
    """




    camera.start_preview()
    camera.capture('/home/pi/Desktop/proj/image123.png')
    image = '/home/pi/Desktop/proj/image123.png'

    gray_img = cv2.imread(image)
    gray_img = np.asarray(gray_img)
    #print(type(gray_img))
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Detecting Lane', gray_img)


    qwer = gray_img
    qwer[qwer>=218] = 230
    qwer[qwer<218] = 30

    """
    return_value, image = cap.read()
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(image, (80, 40))

    qwer = gray_img
    qwer[qwer>=165] = 230
    qwer[qwer<165] = 30
    """
    pre_data = np.asarray(qwer)
    pre_data = pre_data.flatten()

    

    """
    #Turning 2D into 1D
    pre_data = np.asarray(gray_img)
    pre_data = pre_data.flatten()

    qwer = pre_data
    qwer[qwer>=165] = 230
    qwer[qwer<165] = 30
    """

    #cv2.imshow('Detecting Lane', qwer)

    return pre_data
