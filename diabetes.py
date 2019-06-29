#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:57:10 2019

@author: kaustubh
"""

import pandas as pd

df = pd.read_csv('cleaned.csv')

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Outcome'] = le.fit_transform(df['Outcome'])

print('Correlation is \n',df.corr())

import seaborn as sns

sns_plot = sns.heatmap(df.corr(),cmap='coolwarm',linewidth=0.5,annot=True,annot_kws = {"size":5})
fig = sns_plot.get_figure()
fig.savefig('heat.png')
#import matplotlib.pyplot as plt

