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

ocr(1013,1026,291,870,119,131)