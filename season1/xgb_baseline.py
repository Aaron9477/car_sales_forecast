# coding=UTF-8
import os
import gc
import time
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
import operator
import gc
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import lightgbm as lgb

traindata = pd.read_csv('[new] yancheng_train_20171226.csv')
testdata = pd.read_csv('yancheng_testA_20171225.csv')

data = traindata.groupby(['sale_date','class_id'])['sale_quantity'].sum().reset_index()
data.columns=['sale_date','class_id','sum_quan']


def get_traindata(month):
    data = traindata.groupby(['sale_date', 'class_id'])['sale_quantity'].sum().reset_index()
    data.columns = ['sale_date', 'class_id', 'sum_sale']
    data = data[(data.sale_date==2017*100+month)]
    return data

def cal_day(month,k):
    if(month-k>=1):
        return 2017*100+(month-k)
    else:
        return 2016*100+(12+(month-k-1))
def get_feature1(month,k):
    all = data[['class_id']].drop_duplicates()

    for i in range(1,k+1):
        pre_day = cal_day(month, i)
        print(pre_day)
        data_t = data[data.sale_date==pre_day]
        data_t.columns=['sale_date','class_id','pre_'+str(i)]
        data_t = data_t[['class_id','pre_'+str(i)]]
        all = pd.merge(all,data_t,how='left',on=['class_id'])

    return all


traindata_month_7 = get_traindata(7)
traindata_month_8 = get_traindata(8)
traindata_month_9 = get_traindata(9)
traindata_month_10 = get_traindata(10)
traindata_month_11  = testdata

traindata_month_feature_7 = pd.DataFrame(get_feature1(7,12))#每个月前12个月的销量序列
traindata_month_feature_8 = pd.DataFrame(get_feature1(8,12))
traindata_month_feature_9 = pd.DataFrame(get_feature1(9,12))
traindata_month_feature_10 = pd.DataFrame(get_feature1(10,12))
traindata_month_feature_11 = pd.DataFrame(get_feature1(11,12))



traindata_month_7 = pd.merge(traindata_month_7,traindata_month_feature_7,how='left',on=['class_id'])#每个月当月销量以及前12个月的销量的序列
traindata_month_8 = pd.merge(traindata_month_8,traindata_month_feature_8,how='left',on=['class_id'])
traindata_month_9 = pd.merge(traindata_month_9,traindata_month_feature_9,how='left',on=['class_id'])
traindata_month_10 = pd.merge(traindata_month_10,traindata_month_feature_10,how='left',on=['class_id'])
traindata_month_11 = pd.merge(traindata_month_11,traindata_month_feature_11,how='left',on=['class_id'])


traindata_month = pd.concat([traindata_month_9])
feature=[]
for i in range(1,13):
    feature.append('pre_'+str(i))

train1 = xgb.DMatrix(traindata_month[feature], label=traindata_month['sum_sale'])
test1 = xgb.DMatrix(traindata_month_10[feature], label=traindata_month_10['sum_sale'])
param = {
    'objective': 'reg:linear',#regression  linear
    'eta': 0.05,                    #just like learning rate通常最后设置eta为0.01~0.2
    'colsample_bytree': 0.8,#在建立树时对特征随机采样的比例
    'subsample': 0.8,#用于训练模型的子样本占整个样本集合的比例
    'silent': 0,#取0时表示打印出运行时信息
    'verbose_eval': True,
    'eval_metric': 'rmse',#校验数据所需要的评价指标，不同的目标函数将会有缺省的评价指标（rmse for regression, and error for classification, mean average precision for ranking）
    'seed': 666666,#随机数的种子
    'max_depth': 3,#树的最大深度,树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）

}


evallist = [(train1, 'train'), (test1, 'eval')]

bst = xgb.train(param, train1, 1000, evallist,verbose_eval=30,early_stopping_rounds=30)
importance = bst.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
print(importance)


traindata_month_10['pre'] = bst.predict(test1)
traindata_month_10.to_csv('watch.csv',index = False)

train1 = xgb.DMatrix(traindata_month_10[feature], label=traindata_month_10['sum_sale'])
test1 = xgb.DMatrix(traindata_month_11[feature])
evallist = [(train1, 'train')]
bst = xgb.train(param, train1, 55, evallist,verbose_eval=10,early_stopping_rounds=30)
traindata_month_11['predict_quantity'] = bst.predict(test1)
traindata_month_11[['predict_date','class_id','predict_quantity']].to_csv('result.csv',index = False)

