#召回率、准确率计算




a=409
c=378


e=129



b=500   
d=500
f=131



"计算 coal"
TP_coal=a
FP_coal=b-a
FN_coal=d+f-c-e
TN_coal=c+e

Precision_coal=TP_coal/(TP_coal+FP_coal)
Recall_coal=TP_coal/(TP_coal+FN_coal)

F_coal=(2*Precision_coal*Recall_coal)/(Precision_coal+Recall_coal)

# FPR_coal=(FP_coal/(FP_coal+TN_coal))
# TPR_coal=(TP_coal/(TP_coal+FN_coal))

print("coal","\n",Precision_coal,"\n",Recall_coal,"\n",F_coal)

TP_rock=c
FP_rock=d-c
FN_rock=b+f-a-e
TN_rock=a+e
# FPR_rock=(FP_rock/(FP_rock+TN_rock))
# TPR_rock=(TP_rock/(TP_rock+FN_rock))

Precision_rock=TP_rock/(TP_rock+FP_rock)
Recall_rock=TP_rock/(TP_rock+FN_rock)
F_rock=(2*Precision_rock*Recall_rock)/(Precision_rock+Recall_rock)


print("rock","\n",Precision_rock,"\n",Recall_rock,"\n",F_rock)


"计算 _coal_rock"
TP_coal_rock=e
FP_coal_rock=f-e
FN_coal_rock=b+d-a-c
TN_coal_rock=a+c

Precision_coal_rock=TP_coal_rock/(TP_coal_rock+FP_coal_rock)
Recall_coal_rock=TP_coal_rock/(TP_coal_rock+FN_coal_rock)
F_coal_rock=(2*Precision_coal_rock*Recall_coal_rock)/(Precision_coal_rock+Recall_coal_rock)
print("_coal_rock","\n",Precision_coal_rock,"\n",Recall_coal_rock,"\n",F_coal_rock)

# FPR_coal_rock=(FP_coal_rock/(FP_coal_rock+TN_coal_rock))
# TPR_coal_rock=(TP_coal_rock/(TP_coal_rock+FN_coal_rock))
# print("()",(FPR_rock,TPR_rock),(FPR_coal,TPR_coal),(FPR_coal_rock,TPR_coal_rock))

