#***************************************************************
#------  NGUYEN TAN HUNG    MSSV: 16141175  --------------------
#------  NGUYEN MINH TUAN   MSSV: 16141330  --------------------
#------  PHAM VAN LONG      MSSV: 16141194  --------------------
#------  DAO VAN BANG       MSSV: 16141113  --------------------
#***************************************************************

import os
import cv2
from keras import  models
import numpy as np
import pickle
from time import time

path = "test_images"
classes = []
with open('file_name','rb') as pickle_in:
    classes = pickle.load(pickle_in)
print(classes)
model = models.load_model('cifar10_cnn.model')
LDR = os.listdir(path)
for ldr in LDR:
    try:
        start_time = time()
        img_path = os.path.join(path,ldr)
        img_brg = cv2.imread(img_path,1)
        img = cv2.cvtColor(img_brg,cv2.COLOR_BGR2GRAY)/255
        img = cv2.resize(img,(32,32))
        img = img.reshape(1,32,32,1)
        pred = model.predict(img)
        end_time = time()
        tot_time = end_time - start_time
        pred = int(np.argmax(pred,axis=1))
        print('{:<30}{:<10}--time:  {:<15}'.format('Predicted %s as:'%ldr,classes[pred],'%.5f s'%tot_time))
        img_brg = cv2.resize(img_brg,(512,512))
        cv2.putText(img_brg, classes[pred],(20,60),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),3,cv2.LINE_4)
        cv2.imshow('Input image', img_brg)
        cv2.waitKey()
        if  cv2.waitKey(100) == ord(' '):
            break
    except:
        pass

