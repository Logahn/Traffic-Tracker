#!/usr/bin/env python3
from tkinter.ttk import Frame
import cv2
import numpy as np
from PIL import Image
import os
import  random


font = cv2.FONT_HERSHEY_PLAIN
trained_faces_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
trained_palm_data = cv2.CascadeClassifier("palm.xml")

# #* Choose image to detect
# img = cv2.imread("images/multiple.png")

# #* Convert the data to greyscale
# greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #* Detect faces
# face_coordinates = trained_faces_data.detectMultiScale(greyscaled_img)

# #* Draw rectangles around faces
# colors = ((256,0,0), (0,256,0), (0,0,256))

# for (x, y, w, h) in face_coordinates:
#     color = random.choice(colors)
#     cv2.rectangle(img, (x, y),(x+w, y+h), color, 2)

# cv2.imshow("Test data", img)
# cv2.waitKey()


#* Capture video from webcam
webcam = cv2.VideoCapture(0)
#* Capture from video 
#* webcam = cv2.VideoCapture("file_path")


show = True
#* Iterate forever over frames
while show:

    #* Read current frame
    successful_frame_read, frame = webcam.read() 

    #* Convert the data to greyscale
    greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = trained_faces_data.detectMultiScale(
        frame,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
        )
    # #* Detect faces
    face_coordinates = trained_faces_data.detectMultiScale(greyscaled_frame)
    palm_coordinates = trained_palm_data.detectMultiScale(greyscaled_frame)

     #* Draw rectangles around faces
    colors = ((256,0,0), (0,256,0), (0,0,256))  

    for (x, y, w, h) in face_coordinates:
        color = random.choice(colors)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, str("Face"), (x+5,y-5), font,1,color,1)
        #cv2.putText(frame,str("Confidence = 86%"), (x+5,y+h-5),font,1,(255,255,0),1) 

   
    for (x, y, w, h) in palm_coordinates:
        color = random.choice(colors)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, str("Palm"), (x+5,y-5), font,1,color,1)
  

    cv2.imshow("Test data", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        show = False

webcam.release()




#print("code completed")