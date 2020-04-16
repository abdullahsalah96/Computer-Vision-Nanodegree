import cv2
import dlib
import torch
from model import Net
import numpy as np

net = Net()

net.load_state_dict(torch.load('saved_models/face_model.pt'))
net.eval()

# load in color image for face detection
cap = cv2.VideoCapture(0)
while True:
    _, image = cap.read()
    image = cv2.resize(image, (640,480))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # # load in a haar cascade classifier for detecting frontal faces
    # face_cascade = cv2.CascadeClassifier("/Users/abdallaelshikh/Documents/Computer Vision Nanodegree/Facial Keypoints/detector_architectures/haarcascade_frontalcatface.xml")
    # # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    image_with_detections = image.copy()
    image_resized = image.copy()
    out = image.copy()
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 0)
    if True:
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            pad = 120
            cv2.rectangle(image_with_detections,(x-pad,y-pad),(x+w+pad,y+h+pad),(255,0,0),3)
            out = image_with_detections[y-pad:y+h+pad, x-pad:x+w+pad]
            roi = gray[y-pad:y+h+pad, x-pad:x+w+pad]
            image_resized = image.copy()[y-pad:y+h+pad, x-pad:x+w+pad]
            roi = roi/255.0
            roi = cv2.resize(roi, (224,224))
            image_resized = cv2.resize(image_resized, (224,224))
            roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
            roi = np.transpose(roi, (2, 0, 1))
            roi = torch.from_numpy(roi)
            roi = roi.type(torch.FloatTensor)
            roi = roi.unsqueeze(0)
            prediction = net(roi)
            prediction = prediction.view( 68, -1)
            prediction = prediction.data
            prediction = prediction.numpy()
            prediction = prediction*50.0 + 100
            # prediction[:0] = prediction[:0]*640/224
            # prediction[:1] = prediction[:1]*480/224
            # print(prediction[:,1].shape)
            for pt in prediction:
                cv2.circle(image_resized, (int(pt[0]), int(pt[1])), 1, (0,200,0),1)
                pt[0] = pt[0]*w/224.0
                pt[1] = pt[1]*h/224.0
                cv2.circle(out, (int(pt[0]), int(pt[1])), 1, (0,200,0),1)

    cv2.imshow('img', image_with_detections)
    cv2.imshow('resized', image_resized)
    cv2.imshow('out', out)
    if(cv2.waitKey(1) == 'q'):
        break       



            
        