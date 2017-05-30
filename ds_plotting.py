from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score

from treeinterpreter import treeinterpreter as ti

def plot_roc(y_test, y_score, ax = None):
    if not ax:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    auc = roc_auc_score(y_test, y_score)
    fpr,tpr,thresh = roc_curve(y_test, y_score)
    ax.plot([0, 1], [0, 1], linestyle='dashed', color='grey')
    ax.plot(fpr, tpr)
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.02)
    plt.title('ROC')
    plt.text(0.85, 0.03, 'AUC: {:1.3f}'.format(auc))


def plot_importances(cl, column_names, n_features=10, ax=None, error_bars = True):
    df_imp = pd.DataFrame({'features': column_names,
                           'importances': cl.feature_importances_})
    errors = np.std([tree.feature_importances_ for tree in cl.estimators_], axis=0)
    df_imp_sub = df_imp.set_index('features').sort_values('importances').tail(n_features)
    if error_bars:
        df_errors = pd.DataFrame({'features': column_names,
                                  'importances': errors})
        df_err_sub = df_errors.set_index('features').ix[df_imp_sub.index]
    else:
        df_err_sub = None
    ax = df_imp_sub.plot(kind='barh', width=.7, legend=False, ax=ax, xerr=df_err_sub, ecolor='g')
    for i,t in enumerate(df_imp_sub.index.tolist()):
        t = ax.text(0.001, i-.06,t)
        t.set_bbox(dict(facecolor='white', alpha=0.4, edgecolor='grey'))
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_title('Feature Importances')
    ax.set_xlim(0.0)
    ax.set_xlabel('importance')
    return df_imp_sub

def plot_contrib(contribs, ax, label, pred):
    ax = contribs.plot(kind='barh', width=0.7, ax=ax)
    title = 'id: {}'.format(contribs.name)
    title += '; label: {}'.format(label)
    title += '; pred: {:2.2f}'.format(pred)
    ax.set_title(title)
    return ax

def plot_contrib_values(feature_values, ax):
    feature_values = feature_values
    x_coord = ax.get_xlim()[0]
    for y_coord,(f,v) in enumerate(feature_values.iteritems()):
        t = ax.text(x_coord,y_coord,'{:2.2f}'.format(v))
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='blue'))
