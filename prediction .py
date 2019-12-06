# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:29:51 2019

@author: one
"""
import sys
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt 
import tensorflow as tf

# 图片指定尺寸
target_size = (256 , 256) #fixed size for InceptionV3 architecture
# 预测函数
# 输入：model，图片，目标尺寸
# 输出：预测predict
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)
 
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  preds = model.predict(x)
  return preds[0] 

# 画图函数
# 预测之后画图，这里默认是猫狗，当然可以修改label 

labels = ("coal", "coal and rock","rock")
model = load_model('E:\\notebook\\revise\\ex_3.hdf5')
# 本地图片

def path_predict(path):
    filecount=0
    import os
    for root,dir,files in  os.walk(path):
        filecount+=len(files)
    print(filecount)
    m=0
    c=0
    k=0
    for i in range(filecount): 
        #img = Image.open('E:\\notebook\\class\\data\\test\\rock/{}.JPG'.format(i))
        img = Image.open(path+"{}.JPG".format(i))
        I = predict(model, img, target_size)
        n=max(I)
        L=labels[I.tolist().index(n)]

        if L==labels[2]:
            m=m+1
        if L==labels[1]:
            c=c+1
        if L==labels[0]:
            k=k+1

        #print ('图片',i,'为',L,'的概率为',n)
    print("coal is",k,"\n"
          "coal_rock is",c,"\n"
          "rock is",m)
    
path_predict("E:\\notebook\\class\\data\\test\\rock\\")
path_predict("E:\\notebook\\class\\data\\test\\coal\\")
path_predict("E:\\notebook\\class\\data\\test\\coal_rock\\")

#In[]
import numpy as np
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
from keras.models import load_model

model = load_model('ex_1.hdf5')
img = Image.open("data\\class\\lable\\coal\\0.jpg")
target_size = (256 , 256) 
if img.size != target_size:
  img = img.resize(target_size)


x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
print(preds[0])


# %%
