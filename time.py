# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:29:51 2019

@author: one
"""

#模型预测用时
import sys
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt 
import tensorflow as tf
import time


# 图片指定尺寸
target_size = (256 , 256) #fixed size for InceptionV3 architecture
# 预测函数
# 输入：model，图片，目标尺寸
# 输出：预测predict
def predict(model, img, target_size):

  if img.size != target_size:
    img = img.resize(target_size)
 
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  preds = model.predict(x)
  return preds[0] 
# 画图函数
# 预测之后画图，这里默认是猫狗，当然可以修改label 

labels = ("coal", "coal and rock","rock")
# model = load_model('E:\\Exercise\\paper\\7\\model_NET\\trainmodel.hdf5')
# 本地图片
model = load_model("ex_1.hdf5")
start_0 = time.clock()
for i in range(100): 
    img = Image.open('E:\\Exercise\\paper\\test\\coal\\{}.JPG'.format(i))
    I = predict(model, img, target_size)
#当中是你的程序
elapsed_0 = (time.clock() - start_0)
print("Time used 0:",elapsed_0)



start_1 = time.clock()
for i in range(100): 
    img = Image.open('E:\\Exercise\\paper\\test\\coal\\{}.JPG'.format(i))
    I = predict(model, img, target_size)
#当中是你的程序
elapsed_1 = (time.clock() - start_1)
print("Time used 1:",elapsed_1)


start_2 = time.clock()
for i in range(100): 
    img = Image.open('E:\\Exercise\\paper\\test\\rock\\{}.JPG'.format(i))
    I = predict(model, img, target_size)
#当中是你的程序
elapsed_2 = (time.clock() - start_2)
print("Time used 2:",elapsed_2)


start_3 = time.clock()
for i in range(100): 
    img = Image.open('E:\\Exercise\\paper\\test\\coal_rock\\{}.JPG'.format(i))
    I = predict(model, img, target_size)
#当中是你的程序
elapsed_3 = (time.clock() - start_3)
print("Time used 3:",elapsed_3)