import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendence'   #The path where we putted our pictures for trainging
images = []                  #where images will store in form of array
classNames = []
myList = os.listdir(path)    #collects all the files in the given path
print(myList)

#import the images one by one using a loop

for cl in myList:                              #collects all class collected from mylist
 curImg = cv2.imread(f'{path}/{cl}')              #reads images
 images.append(curImg)                            #adds images to images[]
 classNames.append(os.path.splitext(cl)[0])       #adds classes to classNames[]
 print(classNames)


#encodes the images and returns a lists
 def findEncodings(images):
     encodeList = []
     for img in images:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       encode = face_recognition.face_encodings(img)[0]
       encodeList.append(encode)
     return encodeList





#Taking Attendence and putting these in another file
 def markAttendance(name):
     with open('Attendance.csv', 'r+') as f:
         myDataList = f.readlines()
         nameList = []
         for line in myDataList:
             entry = line.split(',')
             nameList.append(entry[0])
         if name not in nameList:
             now = datetime.now()
             dtString = now.strftime('%H:%M:%S')
             f.writelines(f'\n{name},{dtString}')

 encodeListKnown = findEncodings(images)
 print('Encoding Complete')

#starting webcam for video capture

cap = cv2.VideoCapture(0)
# #
while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
#
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
             matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
             faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
             # print(faceDis)
             matchIndex = np.argmin(faceDis)

             if matches[matchIndex]:
                 name = classNames[matchIndex].upper()
                 y1, x2, y2, x1 = faceLoc
                 y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                 cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                 markAttendance(name)


        cv2.imshow('Webcam',img)
        cv2.waitKey(1)