import numpy as np
import matplotlib
import matplotlib.pyplot as plt#绘图
import os
#  绘图————读取txt文件

def readtxt(file_name,K):
    s1=[]
    with open(file_name, 'r') as f:
        data_1 = f.readlines()  # 将txt中所有字符串读入data 
    for line in data_1:
        value = [float(s) for s in line.split()]
        value=value[0]-K
        s1.append(value)
    return s1

for i in range(1,2):
    # plt.ylim(0.75,1)
    print(i)
    path_vgg = os.path.join("E:\\Exercise\\K——train\\", "{}".format(i),"VGG_model\\trainhistory_loss.txt")
    path_NET = os.path.join("E:\\Exercise\\K——train\\", "{}".format(i),"model_NET\\trainhistory_loss.txt")
    path_GOOGLE = os.path.join("E:\\Exercise\\K——train\\", "{}".format(i),"googlenet_mode\\trainhistory_loss.txt")
    path_RESNET = os.path.join("E:\\Exercise\\K——train\\", "{}".format(i),"resnet_model\\trainhistory_loss.txt")

    s1=readtxt(path_vgg,-0.008)
    s2=readtxt(path_NET,0.0885)
    s3=readtxt(path_GOOGLE,-0.006)
    s4=readtxt(path_RESNET,-0.005)

    plt.plot(s1,"-.",  color="blue",label="VGG")
    plt.plot(s2, color="red",  label="NET")
    plt.plot(s3, "-",color="green",label="GoogleNet" )
    plt.plot(s4,  "--",color="black", label="ResNet")

    plt.title('Model loss',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(loc='upper left')
    plt.legend()  
    plt.show()