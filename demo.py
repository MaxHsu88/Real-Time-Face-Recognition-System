import os
import sys
import numpy as np
import pandas as pd
import cv2
import dlib
import openface
import joblib
import sklearn
from model import create_model

# 取得已經壓縮完成的 embeddings 資料
X_openface = pd.read_csv('./embeddings/6_people_embedding_openface_norm.csv', delimiter=',', header=None)
y_openface = pd.read_csv('./embeddings/6_people_label_openface_norm.csv', header=None)
label_encoder = joblib.load('./classifiers/label_encoder.pkl')
y_out = label_encoder.transform(y_openface)

# 引入該用的臉部擷取物件
print("\n Preparing for face detection module ......")
predictor_model = "./face predictors/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()                      # 抓出臉孔
face_pose_predictor = dlib.shape_predictor(predictor_model)           # 抓出臉孔輪廓
face_aligner = openface.AlignDlib(predictor_model)                    # 將臉孔輪廓對齊

# 引入壓縮向量的 CNN Model (FaceNet)
# print("\n Loading FaceNet Model ......")
# face_model = InceptionResNetV1()
# face_model.load_weights('facenet_keras_weights.h5')

# 引入壓縮向量的 CNN Model (OpenFace)
print("\n Loading OpenFace Model ......")
openface_model = create_model()
openface_model.load_weights('./pretrained models/nn4.small2.v1.h5')

# 開啟相機
print("\n Opening Webcam ......")
video_capture = cv2.VideoCapture(0)

# 設定擷取影像的尺寸大小
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 若沒有打開, 則打開
if (not video_capture.isOpened()):
    video_capture.open()
    
# 設定辨認閾值
threshold = 0.25

while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 用 face_detector 去找出臉孔位置及特徵
    detected_faces = face_detector(image, 1)

    # 處理每個臉孔的資料
    for face_rect in detected_faces:
        
        # print(face_rect)
        
        # 抓取臉孔資訊並壓縮成 128 維度的向量
        pose_landmarks = face_pose_predictor(image, face_rect)
        alignedFace = face_aligner.align(96, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        norm_alignedFace = cv2.normalize(alignedFace, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_alignedFace = norm_alignedFace[np.newaxis]
        face_description = openface_model.predict(norm_alignedFace)
        
        # 用 L2 Distance 去判斷是誰
        diff = np.subtract(X_openface.to_numpy(), face_description)
        dist = np.sum(np.square(diff), axis=1)
        idx = np.argmin(dist)
        #print(dist[idx])
        if dist[idx] < threshold:
            predict_name = label_encoder.inverse_transform([y_out[idx]])
        else:
            predict_name = "Not Detect Anyone!"
        
        # 畫出辨認的框框 
        x1 = face_rect.left()
        y1 = face_rect.top()
        x2 = face_rect.right()
        y2 = face_rect.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
        cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (80,18,236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(predict_name), (x1 + 10, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    cv2.imshow('Video', frame)
    
    # 按下 'q' 鍵可以關閉 WebCam
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print("\n Closing Windows ......")
# 釋放所有資源
video_capture.release()
cv2.destroyAllWindows()
print("\n Window Successfully Closed!")