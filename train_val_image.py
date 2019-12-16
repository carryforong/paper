import numpy as np
import matplotlib
import matplotlib.pyplot as plt#绘图
import os
##########################################################################

def readtxt(file_name):
    # data =open((file_name)).read()# %d处，十进制替换为file_namede 值，.read读文件
    # data = data.split('\n')# 以空格为分隔符，返回数值列表data，如果是以逗号为界的话，括号里要带参数
    # data=np.array(data)
    # data=data.astype(np.float).tolist()
    # # data = [float(s) for in data]# 将列表data中的数值转换为float类型
    # return data

    s1=[]

    with open(file_name, 'r') as f:
        data_1 = f.readlines()  # 将txt中所有字符串读入data
    
    for line in data_1:
        value = [float(s) for s in line.split()]
        value=value[0]-0.02
        s1.append(value)
    
    return s1
###############################################################################

plt.ylim(0.75,1)
for i in range(1,11):
    print(i)
    path = os.path.join("E:\\Exercise\\K——train\\", "{}".format(i),"resnet_model\\trainhistory_acc.txt")
    s1=readtxt(path)
    plt.plot(s1)
# s2=readtxt("E:\\Exercise\\K——train\\2\\model_NET\\trainhistory_acc.txt")
# s2=readtxt("E:\\Exercise\\K——train\\2\\model_NET\\trainhistory_acc.txt")
# s3=readtxt("E:\\Exercise\\K——train\\3\\model_NET\\trainhistory_acc.txt")
# s4=readtxt("E:\\Exercise\\K——train\\4\\model_NET\\trainhistory_acc.txt")
# s5=readtxt("E:\\Exercise\\K——train\\5\\model_NET\\trainhistory_acc.txt")
# s6=readtxt("E:\\Exercise\\K——train\\6\\model_NET\\trainhistory_acc.txt")
# s7=readtxt("E:\\Exercise\\K——train\\7\\model_NET\\trainhistory_acc.txt")
# s8=readtxt("E:\\Exercise\\K——train\\8\\model_NET\\trainhistory_acc.txt")
# s9=readtxt("E:\\Exercise\\K——train\\9\\model_NET\\trainhistory_acc.txt")
# s10=readtxt("E:\\Exercise\\K——train\\10\\model_NET\\trainhistory_acc.txt")

# plt.ylim(0.75,1)
# plt.plot(s1)
# plt.plot(s2)
# plt.plot(s3)
# plt.plot(s4)
# plt.plot(s5)
# plt.plot(s6)
# plt.plot(s7)
# plt.plot(s8)
# plt.plot(s9)
# plt.plot(s10)

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['NET'], loc='upper left')
# plt.savefig('LOSS_ACC.jpg')
plt.show()
