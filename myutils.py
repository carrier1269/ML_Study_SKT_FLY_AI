import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import myutils as my
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_iris(mode = None):
    iris = pd.read_csv('iris.csv')
    
    df = iris.drop(['Id'], axis = 1).copy()
    
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
       'species']
    
    # if(mode == 'bin'):
        # df = df.loc[df['species'] != 'Iris-virginica']
    
    df['species'] = df['species'].map({
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2})
    
    x = df.iloc[:,:-1] 
    y = df.iloc[:,-1]

    
    return train_test_split(x, y, test_size = 0.2, random_state = 2022)

# 평가지표 metrics
def print_score(y_true, y_pred, average = 'binary'):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average = average)
    rec = recall_score(y_true, y_pred, average = average)
    print('accuraccy:', acc)
    print('precision:', pre)
    print('recall:', rec)
    
def plot_confusion_matrix(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cfm, annot = True, cbar = False)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()