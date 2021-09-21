import cv2
import numpy as np
from keras.models import load_model

MODEL_PATH="C:\\Users\\Balaji\\Documents\\Machine Learning\\Deep Learning\\Machine Learning\\Deep Learning\\Gender Detection\\Gender-Detection-master\\Gender_Detection_vgg16.h5"
model = load_model(MODEL_PATH)
source = cv2.VideoCapture(0) 
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict={0:'MALE',1:'FEMALE'}
color_dict={0:(0,255,0),1:(0,0,255)}
while(True):
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)  
    for (x,y,w,h) in faces:
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()