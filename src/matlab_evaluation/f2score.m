function f = f2score(TP, TN, FP, FN)
    TN;
    precision = TP/(TP+FP);
    recall = TP/(TP + FN);
    f = 5*((precision*recall)/(4*precision+recall));
end




