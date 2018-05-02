function mccs = evaluate(scores, labell)
    mccs = zeros(100,11);
    counter = 0;
    for threshold = 0:0.01:1
        counter = counter + 1;
        TP = 0;
        FN = 0;
        TN = 0;
        FP = 0;
        for i = 1:length(labell)
            if labell(i) == 1
                if scores(i) >= threshold
                    TP = TP + 1;
                else
                    FN = FN + 1;
                end
            else
                if scores(i) >= threshold
                    FP = FP + 1;
                else
                    TN = TN + 1;
                end
            end
        end
        mccs(counter, 1) = threshold;
        mccs(counter, 2) = TP;
        mccs(counter, 3) = FN;
        mccs(counter, 4) = TN;
        mccs(counter, 5) = FP;
        mccs(counter, 6) = mcc(TP, TN, FP, FN);
        mccs(counter, 7) = fscore(TP, TN, FP, FN);
        mccs(counter, 8) = f2score(TP, TN, FP, FN);
        mccs(counter, 9) = ((TP+TN)/(TP+TN+FP+FN));
        mccs(counter, 10) = ((TP)/(TP+FN));
        mccs(counter, 11) = ((TN)/(TN+FP));
    end
end

