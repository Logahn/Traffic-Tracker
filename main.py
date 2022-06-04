#!/usr/bin/env python3
import cv2
import  random

#* Training algo for cars
car_classifier = cv2.CascadeClassifier("modules/cars.xml")

#* Training algo for bikes
bike_classifier = cv2.CascadeClassifier("modules/bike.xml")

#* Training algo for buses
bus_classifier = cv2.CascadeClassifier("modules/bus.xml")

#* Training algo for pedestraints
pedestrian_classifier = cv2.CascadeClassifier("modules/pedestrian.xml")

font = cv2.FONT_HERSHEY_PLAIN


#* Choose image to detect
# data_input = cv2.imread("data/data1.jpg")
data_input = cv2.imread("data/data1.jpeg")

# # #* Convert the data to greyscale
greyscaled_data = cv2.cvtColor(data_input, cv2.COLOR_BGR2GRAY)

# # #* Detect individual property
pedestrian_identify = pedestrian_classifier.detectMultiScale(greyscaled_data)
bike_identify = bike_classifier.detectMultiScale(greyscaled_data)
bus_identify = bus_classifier.detectMultiScale(greyscaled_data)
car_identify = car_classifier.detectMultiScale(greyscaled_data)


# # #* Draw rectangles around faces

car_color = (256,125,0)
bike_color = (0,256,0)
bus_color = (0,0,256)
pedestrian_color = (125,256,0)


for (x, y, w, h) in bus_identify:
    cv2.rectangle(data_input, (x, y),(x+w, y+h), bus_color, 2)
    cv2.putText(data_input, str("Bus"), (x+5,y-5), font,1,bus_color,1)

for (x, y, w, h) in car_identify:
    cv2.rectangle(data_input, (x, y),(x+w, y+h), car_color, 2)
    cv2.putText(data_input, str("Cars"), (x+5,y-5), font,1,car_color,1)

for (x, y, w, h) in bike_identify:
    cv2.rectangle(data_input, (x, y),(x+w, y+h), bike_color, 2)
    cv2.putText(data_input, str("Bike"), (x+5,y-5), font,1,bike_color,1)

for (x, y, w, h) in pedestrian_identify:
    cv2.rectangle(data_input, (x, y),(x+w, y+h), pedestrian_color, 2)
    cv2.putText(data_input, str("Person"), (x+5,y-5), font,1,pedestrian_color,1)

cv2.imshow("Test data1", data_input)

cv2.waitKey()
cv2.destroyAllWindows()

