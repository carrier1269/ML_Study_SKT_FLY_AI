import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import myutils as my
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings(action='ignore') 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

def final_test():
    df = pd.read_csv('test.csv')
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    y=y.map({'STANDING':0,
         'SITTING':1,
         'LAYING':2,
         'WALKING':3,
         'WALKING_DOWNSTAIRS':4,
         'WALKING_UPSTAIRS':5})
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2022)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_s = scaler.transform(x_train)
    y_train = y_train.values
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train_s,y_train)
    clf.score(x_train_s, y_train)
    x_test_s = scaler.transform(x_test)
    y_pred = clf.predict(x_test_s)
    clf.fit(x_test_s, y_pred)
    print(clf.score(x_test_s, y_pred)*100,'%')
    
    from sklearn.svm import LinearSVC

    names = ['liner', 'linearSVC', 'Poly', 'rbf', 'knn-5']

    models = [
        SVC(kernel = 'linear', C=1),
        LinearSVC(C=1, max_iter=1000),
        SVC(kernel = 'poly', degree=3),
        SVC(kernel = 'rbf', C=1, gamma=0.7),
        KNeighborsClassifier(n_neighbors=5)
    ]

    scores = {}
    for name, model in zip(names, models):
        model.fit(x_train, y_train)
        s = model.score(x_train, y_train)
        print(name, s)
        scores[name] = s
        
    param_range = [0.001, 0.02, 0.1, 1, 10, 100, 100]
    params = [
         {
        'C' : param_range,
        'gamma' : param_range,
        'kernel' : ['rbf'] # 다차원
        },
        {
        'C' : param_range,
        'kernel' : ['linear'] # 선형
        },
        {
        'C' : param_range,
        'degree' : [2, 3],
        'kernel' : ['poly'] # 2차원
        }
    ]
    
    clf = SVC(random_state = 2022)
    gs = GridSearchCV(estimator = clf, # y 추측 데이터값과 x 테스트 데이터 대입
                  param_grid = params,
                  scoring = 'accuracy', # 정확도 측정
                  cv = 3, # 
                  n_jobs=-1, # 학습하는 쓰레드의 수, 최대값인 -1로 최대성능치를 끌어낸다
                  verbose=3
            )
    gs.fit(x_train, y_train)
    print('최종 결과는 ',gs.best_score_*100,'% 입니다.')