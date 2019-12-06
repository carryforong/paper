import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
sys.path.append('..')

DATA_FORMAT='channels_last'
LRN2D_NORM=True
WEIGHT_DECAY=0.0005
  
def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format='channels_last',dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY):
    #l2 normalization
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    if lrn2d_norm:
        #batch normalization
        x=BatchNormalization()(x)

    return x

def inception_module(x,params,concat_axis,padding='same',data_format='channels_last',dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=None):
    (branch1,branch2,branch3,branch4)=params
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None
    #1x1
    pathway1=Conv2D(filters=branch1[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    #1x1->3x3
    pathway2=Conv2D(filters=branch2[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway2=Conv2D(filters=branch2[1],kernel_size=(3,3),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway2)

    #1x1->5x5
    pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)

    #3x3->1x1
    pathway4=MaxPooling2D(pool_size=(2,2),strides=1,padding=padding,data_format=DATA_FORMAT)(x)
    pathway4=Conv2D(filters=branch4[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway4)

    return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)
  
  
  
def NET_2(input_size=(256,256,3)):
  
    inputs=Input(input_size)
    conv1=conv2D_lrn2d(inputs,64,(3,3),1,padding='same',lrn2d_norm=False)
    conv1=conv2D_lrn2d(conv1,64,(3,3),1,padding='same',lrn2d_norm=False)
#     conv1=Conv2D(64,3,activation='relu',padding='same')(inputs)#64*256*256
#     conv1=Conv2D(64,3,activation='relu',padding='same')(conv1)#64*256*256
    pool1=MaxPooling2D(pool_size=(2,2))(conv1) #64*128*128
    pool1=BatchNormalization()(pool1)

    
    conv2=conv2D_lrn2d(pool1,128,(3,3),1,padding='same',lrn2d_norm=False)
    conv2=conv2D_lrn2d(conv2,128,(3,3),1,padding='same',lrn2d_norm=False)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)#128*64*64
    pool2=BatchNormalization()(pool2)

    x=inception_module(pool2,params=[(64,),(96,128),(16,32),(32,)],concat_axis=3)
    pool3=MaxPooling2D(pool_size=(2,2))(x)#256*32*32
    pool3=BatchNormalization()(pool3)

    x=inception_module(pool3,params=[(64,),(96,128),(16,32),(32,)],concat_axis=3)#512*32*32
    pool4=MaxPooling2D(pool_size=(2,2))(x)#256*16*16
    pool4=BatchNormalization()(pool4)

    x=inception_module(pool4,params=[(64,),(96,128),(16,32),(32,)],concat_axis=3)
    pool5=MaxPooling2D(pool_size=(2,2))(x)#256*8*8
    pool5=BatchNormalization()(pool5)
    
    conv=Conv2D(512,1,activation='relu',padding='same')(pool5)
    pool6=AveragePooling2D((8,8))(conv)

    flat=Flatten()(conv)

    dens1=Dropout(0.5)(flat)

    dens=Dense(3,activation='softmax')(dens1)

    model=Model(input=inputs,output=dens)

    return model


def train(train_dir,val_dir):
    # train_dir = 'E:\\notebook\\data\\k=1\\train' 
    # val_dir='E:\\notebook\\data\\k=1\\val' 

    train_datagen =  ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
    val_datagen =  ImageDataGenerator(
    rotation_range=30
    )


    #数据输入及打标签
    train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,class_mode='categorical')

    model=NET_2()

    model.compile(optimizer = Adam(lr = 1e-4),loss="binary_crossentropy",metrics=["accuracy"])  

    model_checkpoint = ModelCheckpoint(train_dir+'model.hdf5', monitor='loss',verbose=1, save_best_only=True)
    history=model.fit_generator(train_generator,steps_per_epoch=50,epochs=200,callbacks=[model_checkpoint],validation_data=val_generator)


    np.savetxt(train_dir+"history_acc.txt", history.history['acc'],fmt='%f',delimiter=',')
    np.savetxt(train_dir+"history_loss.txt", history.history['loss'],fmt='%f',delimiter=',')
    np.savetxt(train_dir+"history_val_acc.txt", history.history['val_acc'],fmt='%f',delimiter=',')
    np.savetxt(train_dir+"history_val_loss.txt", history.history['val_loss'],fmt='%f',delimiter=',')

    print(history.history['acc'])
    print(history.history['loss'])
    print(history.history['val_acc'])
    print(history.history['val_loss'])

# train("E:\\notebook\\data\\k=1\\train","E:\\notebook\\data\\k=1\\val")
# train("E:\\notebook\\data\\k=2\\train","E:\\notebook\\data\\k=2\\val")
train("E:\\notebook\\data\\k=3\\train","E:\\notebook\\data\\k=3\\val")
train("E:\\notebook\\data\\k=4\\train","E:\\notebook\\data\\k=4\\val")
train("E:\\notebook\\data\\k=5\\train","E:\\notebook\\data\\k=5\\val")

# train("E:\\notebook\\data\\k=6\\train","E:\\notebook\\data\\k=6\\val")
# train("E:\\notebook\\data\\k=7\\train","E:\\notebook\\data\\k=7\\val")
# train("E:\\notebook\\data\\k=8\\train","E:\\notebook\\data\\k=8\\val")
# train("E:\\notebook\\data\\k=9\\train","E:\\notebook\\data\\k=9\\val")
# train("E:\\notebook\\data\\k=10\\train","E:\\notebook\\data\\k=10\\val")