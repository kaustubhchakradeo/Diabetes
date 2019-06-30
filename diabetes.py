#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:57:10 2019

@author: kaustubh
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',500)

df = pd.read_csv('cleaned.csv')

Original_df = pd.read_csv('diabetes.csv')
print(Original_df.describe())

Original_df_copy = Original_df.copy(deep = True)
Original_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = Original_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

Original_df_copy['Glucose'].fillna(Original_df_copy['Glucose'].mean(), inplace = True)
Original_df_copy['BloodPressure'].fillna(Original_df_copy['BloodPressure'].mean(), inplace = True)
Original_df_copy['SkinThickness'].fillna(Original_df_copy['SkinThickness'].median(), inplace = True)
Original_df_copy['Insulin'].fillna(Original_df_copy['Insulin'].median(), inplace = True)
Original_df_copy['BMI'].fillna(Original_df_copy['BMI'].median(), inplace = True)



Original_Matrix=pd.plotting.scatter_matrix(Original_df,figsize=(25, 25))
plt.savefig(r'Original_df')

Copy_Matrix=pd.plotting.scatter_matrix(Original_df_copy,figsize=(25, 25))
# sns.pairplot(Original_df_copy,hue = 'Outcome')
plt.savefig(r'Copy_Matrix')


le = preprocessing.LabelEncoder()
df['Outcome'] = le.fit_transform(df['Outcome'])

# print('Correlation is \n',df.corr())



# sns_plot = sns.heatmap(df.corr(),cmap='coolwarm',linewidth=0.5,annot=True,annot_kws = {"size":5})
# fig = sns_plot.get_figure()
# fig.savefig('heat.png')


# ###########  Data Cleaning to be done      ############# #

# 1.Insulin and DiabetesPedigreeFunction needs to be clipped
# 2.Feature Scaling on Glucose
# 3.We can use binning on age or BMI 
# 4.We need to replace the zero values or not?
# 5.Do we need to scale the data?
# 6.Splitting the data for train,test and validate we need to split such that
# 			nos of diabetes patient must not be more than non-diabetes in validation or test data
# 7.