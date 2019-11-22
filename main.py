import net
import data_processing
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


model=net.vgg11_bn()
model.cuda()
cost=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

EPOCH=5
train_data,len_trainset=data_processing.data()

for epoch in range(EPOCH):
    for i, (images, labels) in enumerate(train_data):
    # for images,labels in train_data:
        images=Variable(images).cuda()
        labels=Variable(labels).cuda()

        optimizer.zero_grad()        
        with torch.no_grad():
            outputs=model(images)
        loss=cost(outputs,labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0 :
            print ('Epoch [%d/%d], Iter[%d/%d] Loss. %.4f' %
                (epoch+1, EPOCH, i+1, len_trainset//16, loss.data[0]))

torch.save(model.state_dict(), 'vgg.pkl')