import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model_emotion.h5')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
my_video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
while True:
    ret,frame = my_video.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray,1.3)
    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),8)
        roi_gray = frame_gray[y:y + h, x:x + w]
        crop_image = np.expand_dims(np.expand_dims(cv2.resize(roi_gray,(48,48)),-1),0)
        y_pred = model.predict(crop_image)
        y_pred = np.argmax(y_pred)
        cv2.putText(frame, emotion_dict[y_pred], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

my_video.release()
# cv2.destroyAllWindows()
    # cv2.imshow('output',frame)
    # cv2.waitKey(10)