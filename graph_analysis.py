import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as it

import scipy
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import pandas as pd
import math

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Range1d
from sklearn.cross_validation import train_test_split

def prepare_data_for_machine_learning(betti_data_paths):
    list_of_betti_data = []
    class_indicator = 0

    for betti_data_path in betti_data_paths:

        temp_df = (pd.DataFrame.from_csv(betti_data_path))
        temp_df['TARGET'] = class_indicator
        list_of_betti_data.append(temp_df)
        class_indicator +=1

    df_total = pd.concat(list_of_betti_data)

    df_total.to_csv("test.csv", index = False)

    features = df_total.drop('TARGET',1)
    target = df_total['TARGET']

    return features, target


# employ K-fold cross validation
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem



def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))


def plot_roc_curve(target_test, target_predicted_proba):
    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1])

    roc_auc = auc(fpr, tpr)

    p = figure(title='Receiver Operating Characteristic')
    # Plot ROC curve
    p.line(x=fpr, y=tpr, legend='ROC curve (area = %0.3f)' % roc_auc)
    p.x_range = Range1d(0, 1)
    p.y_range = Range1d(0, 1)
    p.xaxis.axis_label = 'False Positive Rate or (1 - Specifity)'
    p.yaxis.axis_label = 'True Positive Rate or (Sensitivity)'
    p.legend.orientation = "bottom_right"
    show(p)


def train_and_evaluate(clf, X_train, X_test, y_train, y_test,list_of_exp_details, results_save="C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\results\\results.csv"):
    clf.fit(X_train, y_train)

    print("Accuaracy on training set:")
    print(clf.score(X_train, y_train))

    print("Accuaracy on test set:")
    print(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix")
    print(metrics.confusion_matrix(y_test, y_pred))

    cm = (metrics.confusion_matrix(y_test, y_pred))

    cm_df = pd.DataFrame(cm, index=['Predicted Class 0', 'Predicted Class 1'],
                         columns=['Actual Class 0', 'Actual Class 1'])

    #results_data_frame = pd.DataFrame([clf.score(X_train, y_train),clf.score(X_test, y_test) ], columns = ["Accuaracy on training set", "Accuaracy on test set:"])
    #print(results_data_frame)
    #details_of_exp = list_of_exp_details
    results_data_frame = pd.DataFrame(0,index=np.arange(1), columns=  ["Accuaracy on training set", "Accuaracy on test set", "number of clouds", "number of nodes", "dimension of cloud", "class_1 sigma", "class_1 mu","class_2 sigma", "class_2 mu"])
    results_data_frame.ix[0,0] = clf.score(X_train, y_train)
    results_data_frame.ix[0,1] = clf.score(X_test, y_test)
    results_data_frame.ix[0,2] = list_of_exp_details[0]
    results_data_frame.ix[0,3] = list_of_exp_details[1]
    results_data_frame.ix[0,4] = list_of_exp_details[2]
    results_data_frame.ix[0,5] = list_of_exp_details[3]
    results_data_frame.ix[0,6] = list_of_exp_details[4]
    results_data_frame.ix[0,7] = list_of_exp_details[5]
    results_data_frame.ix[0,8] = list_of_exp_details[6]




    results_data_frame.to_csv(results_save,index=False,mode='a')
    print(results_data_frame)

    print(cm)
    return cm

