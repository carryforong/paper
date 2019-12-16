import os
#制作lable和input的映射
def Issubstring(substrlist,str):
    flag=True
    for substr in substrlist:
        if not (substrlist in str):
            flag=False
    return flag

def getfilelist(findpath,flagstr=[]):
    filelist=[]
    filename=os.listdir(findpath)
    if len(filename)>0:
        for fn in filename:
            if len(flagstr)>0:
                if Issubstring(flagstr,fn):
                    fullfilename=os.path.join(findpath,fn)
                    filelist.append(fullfilename)
            else:
                fullfilename=os.path.join(findpath,fn)
                filelist.append(fullfilename)

    if len(filelist)>0:
        filelist.sort()
    
    return filelist



train_txt=open('train.txt','w')

imgfile=getfilelist('E:\\Exercise\\paper\\test2\\coal')#.py文件目录下
for img in imgfile:
    str1=img+' '+'0'+'\n'   #用空格代替转义字符 \t 
    train_txt.writelines(str1)

imgfile=getfilelist('E:\\Exercise\\paper\\test2\\rock')#.py文件目录下
for img in imgfile:
    str1=img+' '+'1'+'\n'   #用空格代替转义字符 \t 
    train_txt.writelines(str1)


imgfile=getfilelist('E:\\Exercise\\paper\\test2\\coal_rock')#.py文件目录下
for img in imgfile:
    str1=img+' '+'2'+'\n'   #用空格代替转义字符 \t 
    train_txt.writelines(str1)


    
train_txt.close()
