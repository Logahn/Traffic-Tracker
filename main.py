#!/usr/bin/env python3
import cv2
import  random

#* Training algo for cars
cars_classifier_file = "modules/cars.xml"
car_classifier = cv2.CascadeClassifier("modules/cars.xml")

#* Training algo for bikes
bike_classifier_file = "modules/bike.xml"
bike_classifier = cv2.CascadeClassifier("bike_classifier_file")

#* Training algo for buses
bus_classifier_file = "modules/bus.xml"
bus_classifier = cv2.CascadeClassifier("bus_classifier_file")

#* Training algo for pedestraints
pedestrians_classifier_file = "modules/pedestrians.xml"
pedestrians_classifier = cv2.CascadeClassifier("pedestrians_classifier_file")

font = cv2.FONT_HERSHEY_PLAIN


#* Choose image to detect
img = cv2.imread("data/data1.jpg")

# # #* Convert the data to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # #* Detect faces
face_coordinates = car_classifier.detectMultiScale(greyscaled_img)

# # #* Draw rectangles around faces
colors = ((256,0,0), (0,256,0), (0,0,256))

for (x, y, w, h) in face_coordinates:
    color = random.choice(colors)
    cv2.rectangle(img, (x, y),(x+w, y+h), color, 2)
    cv2.putText(img, str("Cars"), (x+5,y-5), font,1,color,1)

cv2.imshow("Test data1", img)

cv2.waitKey()


# #* Capture video from webcam
# webcam = cv2.VideoCapture(0)
# #* Capture from video 
# #* webcam = cv2.VideoCapture("file_path")


# show = True
# #* Iterate forever over frames
# while show:

#     #* Read current frame
#     successful_frame_read, frame = webcam.read() 

#     #* Convert the data to greyscale
#     greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

#     faces = trained_faces_data.detectMultiScale(
#         frame,     
#         scaleFactor=1.2,
#         minNeighbors=5,     
#         minSize=(20, 20)
#         )
#     # #* Detect faces
#     face_coordinates = trained_faces_data.detectMultiScale(greyscaled_frame)
#     palm_coordinates = trained_palm_data.detectMultiScale(greyscaled_frame)

#      #* Draw rectangles around faces
#     colors = ((256,0,0), (0,256,0), (0,0,256))  

#     for (x, y, w, h) in face_coordinates:
#         color = random.choice(colors)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, str("Face"), (x+5,y-5), font,1,color,1)
#         #cv2.putText(frame,str("Confidence = 86%"), (x+5,y+h-5),font,1,(255,255,0),1) 

   
#     for (x, y, w, h) in palm_coordinates:
#         color = random.choice(colors)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, str("Palm"), (x+5,y-5), font,1,color,1)
  

#     cv2.imshow("Test data", frame)
#     key = cv2.waitKey(1)

#     if key == 81 or key == 113:
#         show = False

# webcam.release()




#print("code completed")