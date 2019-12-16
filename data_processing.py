import torchvision
import torch
import cv2
import PIL
import numpy as np

#pytorch 数据预处理



#Compose 这个类是用来管理各个transform的
#ToTensor  类是实现：Convert a PIL Image or numpy.ndarray to tensor
#Normalize  类是做数据归一化的
#Resize  类是对PIL Image做resize操作的,此时表示将输入图像的短边resize到这个int数，长边则根据对应比例调整，图像的长宽比不变。如果输入是个(h,w)的序列，h和w都是int，则直接将输入图像resize到这个(h,w)尺寸
#CenterCrop  是以输入图的中心点为中心点做指定size的crop操作，一般数据增强不会采用这个，因为当size固定的时候，在相同输入图像的情况下，N次CenterCrop的结果都是一样的。
#RandomCrop  是crop时的中心点坐标是随机的，并不是输入图像的中心点坐标，因此基本上每次crop生成的图像都是有差异的
#RandomHorizontalFlip  类也是比较常用的，是随机的图像水平(Horizonta)Horizonta翻转，通俗讲就是图像的左右对调
#RandomVerticalFlip  类是随机的图像竖直(Vertical）翻转，通俗讲就是图像的上下对调

#RandomResizedCrop  类主要用到3个参数：size、scale和ratio，总的来讲就是先做crop（用到scale和ratio），再resize到指定尺寸（用到size）
# 做crop的时候，其中心点坐标和长宽是由get_params方法得到的，在get_params方法中主要用到两个参数：scale和ratio，首先在scale限定的数值范
# 围内随机生成一个数，用这个数乘以输入图像的面积作为crop后图像的面积；然后在ratio限定的数值范围内随机生成一个数，表示长宽的比值，根据这两个值就可以得到crop图像的长宽了


#ColorJitter  类也比较常用，主要是修改输入图像的4大参数值：brightness,contrast and saturation，hue，也就是亮度，对比度，饱和度和色度。可以根据注释来合理设置这4个参数。
#RandomRotation 类是随机旋转输入图像
#Grayscale 类是用来将输入图像转成灰度图的

def data():
    path="data\\class\\lable"
    batch_size=16
    workers=3

    train_augmentation = torchvision.transforms.Compose([
                                                        torchvision.transforms.Resize(256),
                                                        torchvision.transforms.RandomCrop(224),
                                                        torchvision.transforms.ToTensor(),
                                                        ])



    #train_augmentationj=接受到的数据为 PIL.Image.Image，一般为RGB；输出为归一化的数组，opencv读入图像为BGR图像（img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)）
    # img_PIL = PIL.Image.open(path).convert('RGB')
    # img1 = train_augmentation(img_PIL)
    # img_1 = img1.numpy()*255
    # img_1 = img_1.astype('uint8')
    # img_1 = np.transpose(img_1, (1,2,0))      
    # cv2.imshow('img_1', img_1)
    # cv2.waitKey()

    trainset = torchvision.datasets.ImageFolder(path,train_augmentation,)


    # # print(trainset.classes) #根据分的文件夹的名字来确定的类别
    # # print(trainset.class_to_idx) #按顺序为这些类别定义索引为0,1...
    # # print(trainset.imgs) #返回从所有文件夹中得到的图片的路径以及其类别


    # #shuffle是否打乱数据
    # #num_workers数据分为几个线程处理，默认为0

    train_loader = torch.utils.data.DataLoader(trainset,
        batch_size = batch_size, shuffle = True,
        num_workers = 0, pin_memory = True)
    return train_loader,len(trainset)