from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

def plotROC(test_labels, pred_score):
    """
    Plots roc curve
    :param pred_labels: List of predicted labels
    :param test_labels: List of true labels
    :return: ROC graph
    """
    roc_auc = roc_auc_score(test_labels, pred_score)
    plt.title('Receiver Operating Characteristic')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, pred_score)
    plt.plot(false_positive_rate, true_positive_rate, color="r", label='AUC* = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def getANNResults(test_labels, pred_score):
    """
    Get number of maligne and benigne predicted labels
    :param labels: Labels
    :return: maligne, benigne : Number of maligne and benigne prediction
    """
    maligne_pred = 0
    benigne_pred = 0
    maligne_test = 0
    benigne_test = 0
    for pred in pred_score:
        if pred == 1:
            maligne_pred+=1
        else:
            benigne_pred+=1
    for label in test_labels:
        if label == 1:
            maligne_test+=1
        else:
            benigne_test+=1
    return maligne_pred, benigne_pred, maligne_test, benigne_test

def getPositiveNegatves(test_labels, pred_score):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(0, len(pred_score)):
        if pred_score[i] == 1 and test_labels[i] == 1:
            TP = TP+1
        elif pred_score[i] == 1 and test_labels[i] == 0:
            FP = FP+1
        elif pred_score[i] == 0 and test_labels[i] == 0:
            TN = TN+1
        elif pred_score[i] == 0 and test_labels[i] == 1:
            FN = FN+1
        else:
            print("ERROR")
    print("TP: ", TP," FP: ",FP," TN: ",TN," FN: ",FN)




def getAUC(test_labels, pred_score):
    """
    Compute AUC of prediction
    :param pred_score:
    :param test_labels:
    :return: roc_auc - AUC of prediction
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, pred_score)
    roc_auc = roc_auc_score(test_labels, pred_score)
    return roc_auc


if __name__ == '__main__':
    #getPositiveNegatves(label, pred)
    #print("AUC = %s" % (getAUC(label, pred)))
    #print("ACC = %s" % (accuracy_score(label, pred)))
    #print("MCC = %s" % (metrics.matthews_corrcoef(label, pred)))
    #plotROC(label, pred)
    label = [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    pred = [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

    plotROC(label, pred)

    print("test")



