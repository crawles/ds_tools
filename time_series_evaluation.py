import enum
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import metrics


class Label(enum.IntEnum):
    DOWN = 0
    STAY = 1
    UP = 2

class ClassificationReporter:
    class_names = ('down', 'stay', 'up')

    def __init__(self, scores: np.ndarray, labels: np.ndarray):
        self.scores = scores
        self.predictions = pd.Series(np.argmax(scores.values, 1), scores.index)
        self.labels = labels

    def _make_confusion_matrix(self, ax):
        df_cm = pd.DataFrame(sklearn.metrics.confusion_matrix(
            self.labels, self.predictions),
                             columns=self.class_names,
                             index=self.class_names)
        ax = sns.heatmap(df_cm, annot=True, cmap="OrRd", ax=ax)
        ax.set_ylabel('TRUE')
        ax.set_xlabel('PREDICTED')
        ax.set_title('Confusion Matrix')
        return ax

    def _accuracy(self, predictions: Union[int, np.ndarray],
                  label: np.ndarray):
        return (predictions == label).sum() / len(label)

    def _compute_accuracies(self, ax):
        accuracies = []
        model_names = ['ml_model', 'all_down', 'all_stay', 'all_up']
        for predictions in (self.predictions, 0, 1, 2):
            accuracies.append(self._accuracy(predictions, self.labels))
        series = pd.Series(accuracies, index=model_names)
        ax = sns.heatmap(series.to_frame().T, annot=True, ax=ax)
        ax.set_title('Accuracy')
        return ax

    
    def report(self):
        fig, ax = plt.subplots(figsize=(10,1))
        self._compute_accuracies(ax)
        fig, ax = plt.subplots(figsize=(5,5))
        self._make_confusion_matrix(ax)

    def roc_curve(self, label: Label):
        plot_roc((self.labels == label).astype(int).values, 
                 self.scores.iloc[:, label])
        
        

def plot_roc(y_test: np.ndarray, y_score: np.ndarray, ax = None):
    if not ax:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    auc = metrics.roc_auc_score(y_test, y_score)
    fpr,tpr,thresh = metrics.roc_curve(y_test, y_score)
    ax.plot([0, 1], [0, 1], linestyle='dashed', color='grey')
    ax.plot(fpr, tpr)
    ax.set_ylabel('True Positive Rate', size=16)
    ax.set_xlabel('False Positive Rate', size=16)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.02)
    plt.title('ROC', size=24)
    plt.text(0.75, 0.03, 'AUC: {:1.3f}'.format(auc), size=24)
    return auc

def permutation_importances(test_Xy, feature_columns, label_name, metric):
    """Computes permutation feature importance."""
    def permute(df: pd.DataFrame, col):
        """Randomly shuffles column in df."""
        permuted = df.copy()
        permuted[col] = np.random.permutation(df[col])
        return permuted

    baseline = metric(test_Xy[feature_columns], test_Xy[label_name])
    imp = []
    for col in feature_columns:
        permuted = permute(test_Xy, col)
        delta = baseline - metric(permuted[feature_columns],
                                  permuted[label_name])
        print('crcr df[col].head(), test_Xy.head()',  test_Xy[col].head(), test_Xy[col].head())
        imp.append(delta)
    return np.array(imp)
