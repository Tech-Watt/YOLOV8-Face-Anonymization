import cvzone
from ultralytics import YOLO
import cv2




video = r'C:\Users\Admin\Desktop\data\neutral/10.mp4'

cap = cv2.VideoCapture(video)
facemodel = YOLO('yolov8n-face.pt')


while cap.isOpened():
    rt, video = cap.read()
    video = cv2.resize(video, (700, 500))
    mainvideo = video.copy()

    face_result = facemodel.predict(video,conf = 0.30)
    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h,w = y2-y1,x2-x1

            cvzone.cornerRect(video,[x1,y1,w,h],l=9,rt=3)
            cvzone.cornerRect(mainvideo, [x1, y1, w, h], l=9, rt=3)

            face = video[y1:y1+h,x1:x1+w]
            face = cv2.blur(face,(30,30))
            video[y1:y1+h,x1:x1+w] = face


    allFeeds = cvzone.stackImages([mainvideo,video],2,0.80)
    cv2.imshow('frame', allFeeds)
    # cv2.imshow('face', face)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()