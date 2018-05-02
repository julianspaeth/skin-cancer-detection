function f = fscore(TP, TN, FP, FN)
    precision = TP/(TP+FP);
    recall = TP/(TP + FN);
    f = (2*(precision*recall)/(precision+recall));
end




