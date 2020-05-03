import numpy as np
import pandas as pd
import cv2
import dlib
import openface

def image_embedding(network, image_size):

    # 將所有在 X_path 裡的照片路徑讀取照片進來, 做Alignment過後壓成 128維 的特徵向量

    predictor_model = "./face predictors/shape_predictor_68_face_landmarks.dat"
    # recognizer_model = "dlib_face_recognition_resnet_model_v1"

    face_detector = dlib.get_frontal_face_detector()                      # 抓出臉孔
    face_pose_predictor = dlib.shape_predictor(predictor_model)           # 抓出臉孔輪廓
    face_aligner = openface.AlignDlib(predictor_model)                    # 將臉孔輪廓對齊
    # face_recognizer = dlib.face_recognition_model_v1(recognizer_model)    # 壓縮臉孔成128維向量

    X = np.zeros((1, 128))
    y = y_pd.values
    delete_index = []

    for idx, path in enumerate(X_paths.values):
        print(path)
            
        image_path = my_dataset_path + path

        # 從圖片路徑讀取資料
        image = cv2.imread(image_path)
        
        if (image is not None):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     # 因為 OpenCV 讀取影像時是以BGR而非傳統RGB的順序

            # 用 face_detector 去找出臉孔位置及特徵
            detected_faces = face_detector(image, 1)

            # 只取用含有1個臉孔的資料
            if (len(detected_faces) == 1):
                face_rect = detected_faces[0]
                pose_landmarks = face_pose_predictor(image, face_rect)
                alignedFace = face_aligner.align(image_size, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                norm_alignedFace = cv2.normalize(alignedFace, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                norm_alignedFace = norm_alignedFace[np.newaxis]
                face_description = network.predict(norm_alignedFace)
                X = np.vstack((X, face_description.reshape(1, -1)))
            else:
                delete_index.append(idx)    # 多出一個臉的資料必須去除掉
        else:
            print("failed")
            delete_index.append(idx)    # 多偵測不到臉部的資料必須去除掉

    X = np.delete(X, 0, axis=0)
    y = np.delete(y, delete_index, axis=0)

    return X, y