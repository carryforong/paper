import itertools
import matplotlib.pyplot as plt
import numpy as np
 #混淆矩阵绘图
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
 
#     print(cm)
    cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=25)
    cb=plt.colorbar( )
    cb.ax.tick_params(labelsize=17)  #设置色标刻度字体大小。
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30,fontsize=17)
    plt.yticks(tick_marks, classes,fontsize=17)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=20)
 
    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)
#     plt.rcParams['savefig.dpi'] = 1000 #图片像素  
    plt.savefig('{}.jpg'.format(title),dpi=200,bbox_inches = 'tight') 
    plt.show()
 
NET = np.array([
    [409, 2, 6 ],
    [0, 91, 8 ],
    [5, 7, 86 ],
])

NET_wzq = np.array([
    [72, 9, 8 ],
    [5, 78, 21 ],
    [23, 13, 71 ],
])

NET_wyh= np.array([
    [89, 2, 8 ],
    [2, 84, 11 ],
    [9, 14, 81 ],
])

VGG = np.array([
    [94, 6, 11 ],
    [0, 77, 16 ],
    [6, 17, 73 ],
])
GoogLeNet = np.array([
    [92, 2, 4 ],
    [1, 85, 14 ],
    [7, 13, 82 ],
])
ResNet = np.array([
    [90, 4, 11 ],
    [3, 89, 10 ],
    [7, 7, 79 ],
])


class_names = ['coal', 'rock', 'coal and rock']
 
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
 
# Plot normalized confusion matrix

plot_confusion_matrix(NET, classes=class_names, normalize=True,title='Enhanced data NET')
plot_confusion_matrix(NET_wzq, classes=class_names, normalize=True,title='Raw data NET')