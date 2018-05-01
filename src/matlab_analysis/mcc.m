function mcc = mcc(TP, TN, FP, FN)
    zahler = TP*TN - FP*FN;
    nenner = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    mcc = zahler/nenner;
end




