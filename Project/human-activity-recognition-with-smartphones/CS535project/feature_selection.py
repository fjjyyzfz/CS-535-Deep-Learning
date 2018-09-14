"""
CS 535 Deep Learning Final Project
Author: Zhou Fang,Jiale Liu 
Final Update: 03/20/2018
Discription: ksfeature package should be installed first. 
skfeature package is downloaded from ASU feature selection web.
"""

from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.statistical_based import f_score



if __name__ == "__main__":
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    train_label = train_data['Activity']
    test_label = test_data['Activity']

    train_x = np.array(train_data.drop(['subject', 'Activity'], axis=1))
    test_x = np.array(test_data.drop(['subject', 'Activity'], axis=1))
    encoder = preprocessing.LabelEncoder()
    encoder.fit(train_label)
    classes = list(encoder.classes_)
    train_y = np.array(encoder.transform(train_label))
    test_y = np.array(encoder.transform(test_label))
    print("start feature selection")
    index = MIFS.mifs(train_x, train_y, n_selected_features=400)
    #score = f_score.f_score(train_x, train_y)
    #index = f_score.feature_ranking(score)
    print("end feature selection")

    index_select = sorted(index[:400])
    file = open("selected feature f_score.txt", 'w')
    #file = open("selected feature MIFS.txt", 'w')
    for i in index_select:
        file.write("%d " %(i))
    file.close()
    
