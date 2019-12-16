def ocr(a,b,c,d,e,f):
    "计算rock"
    TP_rock=a
    FP_rock=b-a
    FN_rock=d+f-c-e
    TN_rock=c+e
    FPR_rock=(FP_rock/(FP_rock+TN_rock))
    TPR_rock=(TP_rock/(TP_rock+FN_rock))


    "计算 coal"
    TP_coal=c
    FP_coal=d-c
    FN_coal=b+f-a-e
    TN_coal=a+e

    FPR_coal=(FP_coal/(FP_coal+TN_coal))
    TPR_coal=(TP_coal/(TP_coal+FN_coal))
    "计算 coal_rock"
    TP_coal_rock=e
    FP_coal_rock=f-e
    FN_coal_rock=b+d-a-c
    TN_coal_rock=a+c

    FPR_coal_rock=(FP_coal_rock/(FP_coal_rock+TN_coal_rock))
    TPR_coal_rock=(TP_coal_rock/(TP_coal_rock+FN_coal_rock))
    print((FPR_rock,TPR_rock),(FPR_coal,TPR_coal),(FPR_coal_rock,TPR_coal_rock))

    return (FPR_rock,TPR_rock),(FPR_coal,TPR_coal),(FPR_coal_rock,TPR_coal_rock)
#0
# rock_0,coal_0,coal_rock_0=ocr(1026,1026,870,870,131,131)
# #0.1
# rock_1,coal_1,coal_rock_1=ocr(1026,1026,625,870,127,131)
# #0.2
# rock_2,coal_2,coal_rock_3=ocr(1024,1026,575,870,125,131)
# #0.3
# rock_3,coal_3,coal_rock_3=ocr(1024,1026,545,870,125,131)
# #0.4
# rock_4,coal_4,coal_rock_4=ocr(1024,1026,511,870,125,131)
# #0.5
# rock_5,coal_5,coal_rock_5=ocr(1021,1026,480,870,125,131)
# #0.6
# rock_6,coal_6,coal_rock_6=ocr(1019,1026,453,870,124,131)
# #0.7
# rock_7,coal_7,coal_rock_7=ocr(1017,1026,413,870,124,131)
# #0.8
# rock_8,coal_8,coal_rock_8=ocr(1016,1026,364,870,121,131)
# #0.9
# rock_9,coal_9,coal_rock_9=ocr(1013,1026,191,870,119,131)
# #1
# rock_10,coal_10,coal_rock_10=ocr(0,1026,0,870,0,131)


rock_0,coal_0,coal_rock_0=ocr(1026,1026,870,870,131,131)
#0.1
rock_1,coal_1,coal_rock_1=ocr(1026,1026,756,870,127,131)
#0.2
rock_2,coal_2,coal_rock_2=ocr(1024,1026,754,870,125,131)
#0.3
rock_3,coal_3,coal_rock_3=ocr(1024,1026,745,870,125,131)
#0.4
rock_4,coal_4,coal_rock_4=ocr(1024,1026,723,870,125,131)
#0.5
rock_5,coal_5,coal_rock_5=ocr(1021,1026,715,870,125,131)
#0.6
rock_6,coal_6,coal_rock_6=ocr(1019,1026,706,870,124,131)
#0.7
rock_7,coal_7,coal_rock_7=ocr(1017,1026,699,870,124,131)
#0.8
rock_8,coal_8,coal_rock_8=ocr(1016,1026,698,870,121,131)
#0.9
rock_9,coal_9,coal_rock_9=ocr(1013,1026,678,870,119,131)
#1
rock_10,coal_10,coal_rock_10=ocr(0,1026,0,870,0,131)

import matplotlib.pyplot as plt  
import random

plt.ylim(0,1)
plt.xlim(0,1)


x1 = [rock_1[0],rock_2[0],rock_3[0],rock_4[0],rock_5[0],rock_6[0],rock_7[0],rock_8[0],rock_9[0]]
y1 = [rock_1[1],rock_2[1],rock_3[1],rock_4[0],rock_5[1],rock_6[1],rock_7[1],rock_8[1],rock_9[1]] 

x2 = [coal_1[0],coal_2[0],coal_3[0],coal_4[0],coal_5[0],coal_6[0],coal_7[0],coal_8[0],coal_9[0]]
y2 = [coal_1[1],coal_2[1],coal_3[1],coal_4[0],coal_5[1],coal_6[1],coal_7[1],coal_8[1],coal_9[1]]  


x3 = [coal_rock_1[0],coal_rock_2[0],coal_rock_3[0],coal_rock_4[0],coal_rock_5[0],coal_rock_6[0],coal_rock_7[0],coal_rock_8[0],coal_rock_9[0]]
y3 = [coal_rock_1[1],coal_rock_2[1],coal_rock_3[1],coal_rock_4[0],coal_rock_5[1],coal_rock_6[1],coal_rock_7[1],coal_rock_8[1],coal_rock_9[1]]  


# x=(x1+x2+x3)/3
# y=(y1+y2+y3)/3



plt.plot(x1, y1,  color='r',markerfacecolor='blue',marker='o')
plt.plot(x2, y2,  color='g',markerfacecolor='red',marker='o')
plt.plot(x3, y3,  color='b',markerfacecolor='black',marker='o')


# for a, b in zip(x1, y1):  
#     # plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10) 
#     plt.plot(a,b)


# for a, b in zip(x2, y2):
#     # plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10) 
#     plt.plot(a,b)

# for a, b in zip(x3, y3):  
#     # plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10) 
#     plt.plot(a,b)

# for a, b in zip(x, y):  
#     # plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10) 
#     plt.plot(a,b)

plt.legend()  
plt.show()