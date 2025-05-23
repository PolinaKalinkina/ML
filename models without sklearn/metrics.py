
def roc_auc_score(y_true, y_prob):

    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]


    tpr = [0]
    fpr = [0]
    auc = 0


    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos

    prev_prob = None
    tp = fp = 0

    for i in range(len(y_prob_sorted)):

        if y_prob_sorted[i] != prev_prob:
            tpr.append(tp / n_pos if n_pos != 0 else 0)
            fpr.append(fp / n_neg if n_neg != 0 else 0)
            prev_prob = y_prob_sorted[i]


        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1


    tpr.append(1)
    fpr.append(1)

    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return auc, tpr, fpr

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / (predicted_positives + 1e-10)

def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / (actual_positives + 1e-10)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + 1e-10)
