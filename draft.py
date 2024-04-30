import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Input, Activation
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from math import dist
from sklearn.feature_selection import SelectKBest, chi2
import sys

for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))

# df = pd.DataFrame({
#     'Maths':[6,8,6,14.5,14,11,5.5,13,9],
#     'Sciences':[6,8,7,14.5,14,10,7,12.5,9.5],
#     'Francais':[5,8,11,15.5,12,5.5,14,8.5,12.5],
#     'Latin':[5.5,8,9.5,15,12,7,11.5,9.5,12],
#     'Musique':[8,9,11,8,10,13,10,12,18]
# })
# coor1 = df.corr()
# print(coor1)
# X = df.values
# print(X)
# Z = StandardScaler().fit_transform(X)
# print(Z.T)
# covariance_matrix = np.cov(Z.T)
# print('============协方差矩阵============')
# print(covariance_matrix)


column_list = ['duration','protocol_type','service','flag','src_bytes',
              'dst_bytes','land','wrong_fragment','urgent','hot',
              'num_failed_logins', 'logged_in', 'num_compromised','root_shell','su_attempted',
              'num_root','num_file_creation','num_shells','num_access_files','num_outbound_cmds',
              'is_host_login','is_guest_login','count','srv_count','serror_rate',
              'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
              'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
              'dst_host_srv_rerror_rate','label','difficulty_level']


# df_train = pd.read_table('D:/IDdataset/NSL-KDD/KDDTrain+.txt',delimiter=',',names=column_list)
# df_test = pd.read_table('D:/IDdataset/NSL-KDD/KDDTest+.txt',delimiter=',',names=column_list)
#
# subclass_label = df_train['label']
# print(subclass_label)
# subclass_label.to_csv('D:/IDdataset/label/subclass_train_label.csv',header=0,index=None)

# df = pd.read_csv('D:/IDdataset/label/no_novelty_test_sublabel_numerical.csv',header=0)
# df['numerical'] = df['label'].map({"back":1,
#                                    "buffer_overflow":2,
#                                    "ftp_write":3,
#                                    "guess_passwd":4,
#                                    "imap":5,
#                                    "ipsweep":6,
#                                    "land":7,
#                                    "loadmodule":8,
#                                    "multihop":9,
#                                    "neptune":10,
#                                    "nmap":11,
#                                    "normal":12,
#                                    "perl":13,
#                                    "phf":14,
#                                    "pod":15,
#                                    "portsweep":16,
#                                    "rootkit":17,
#                                    "satan":18,
#                                    "smurf":19,
#                                    "spy":20,
#                                    "teardrop":21,
#                                    "warezclient":22,
#                                    "warezmaster":23})
# df.to_csv('D:/IDdataset/label/no_novelty_test_sublabel_numerical.csv',header=True,index=False)
# df = pd.read_table('D:/IDdataset/reprocess-524/concat.txt',delimiter=',',names=column_list)
# df.to_csv('D:/IDdataset/reprocess-524/concat.csv',header=True,index=False)
onehot_concat_df = pd.read_csv('D:/IDdataset/processedNSL/concat_onehot_ordered.csv', header=0)
onehot_concat_df = onehot_concat_df.drop("difficulty_level",axis=1)

X = onehot_concat_df.iloc[:,0:129]
y = onehot_concat_df.iloc[:,129:]
skb = SelectKBest(score_func=chi2, k=40)
# print(new_X.shape)
newX = skb.fit_transform(X,y)
# new_concat_df.to_csv('D:/IDdataset/processedNSL/new_X_concat.csv',index=False,header=False)

# df_scores = pd.DataFrame(skb.scores_)
# df_columns = pd.DataFrame(X.columns)
# df_feature_scores = pd.concat([df_columns, df_scores], axis=1)
# df_feature_scores.columns = ['Feature', 'Score']
#
# sss = df_feature_scores.sort_values(by='Score', ascending=False)
# print(sss)

