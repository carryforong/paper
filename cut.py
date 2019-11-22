#In[1]
import glob
import cv2
#filepath=glob.glob('1.jpg')
count=0
for i in range (0,36):
    img= cv2.imread('E:/Exercise/数据/新建文件夹/{}.jpg'.format(i))
    height=len(img)
    width=len(img[0])
    n=0
     


    for x in range (0,height,504):
        for y in range (0,width,504):
            img_cut = img[x:x + 504, y:y + 504]
            s = "%05d" % count
            print(s)
            save_dir = "E:/Exercise/数据/新建文件夹/out/{}.jpg".format(s)
            count += 1
            cv2.imwrite(save_dir, img_cut)
    count+=100






#I[2]
import PIL.Image as Image
import os
 
IMAGES_PATH = 'out/'  # 图片集地址
IMAGES_FORMAT = ['.jpg']  # 图片格式
IMAGE_SIZE = 500  # 每张小图片的大小
IMAGE_ROW = 8  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 12  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'image/final.jpg'  # 图片转换后的地址
 
# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
 
# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")
 
# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW+1 ):
        for x in range(1, IMAGE_COLUMN+1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH) # 保存新图

image_compose() #调用函数


#In[3]
print("好好学习")