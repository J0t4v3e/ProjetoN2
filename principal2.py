import dlib  
import cv2  
import matplotlib.pyplot as plt  
import numpy as np 


image_path = "lucas2.jpg"
image = cv2.imread(image_path)


if image is None:
    raise FileNotFoundError(f"Imagem não encontrada")


image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  


image = cv2.GaussianBlur(image, (5, 5), 0) 


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


detector_hog = dlib.get_frontal_face_detector()#detector hog que vem nativamento da bibliotaca dlib


profile_cascade = cv2.CascadeClassifier(r"C:\Program Files\opencv\data\haarcascade_profileface.xml")#importante fazer o download do xml ja treinado


faces_front = detector_hog(gray_image, 1)  


faces_profile_right = profile_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=4,   
    minSize=(30, 30),
    maxSize=(800, 800)  


flipped_gray = cv2.flip(gray_image, 1)  
faces_profile_left = profile_cascade.detectMultiScale(
    flipped_gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
    maxSize=(800, 800)
)


for face in faces_front:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()  
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  


for (x, y, w, h) in faces_profile_right:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  


for (x, y, w, h) in faces_profile_left:
    
    x = gray_image.shape[1] - x - w  
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  


plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
plt.axis('off')  
plt.show()  
