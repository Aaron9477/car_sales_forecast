import pandas as pd
from tqdm import tqdm

DIR = '../data/'

test_behavior = pd.read_csv(DIR + 'yancheng_train_20171226.csv', usecols=[0,1,2])
print('the shape of train data is ', test_behavior.shape)

all_time_axis=[201201+i for i in range(12)] # 月份编号
for i in range(12):
   all_time_axis.append(201301+i)
for i in range(12):
   all_time_axis.append(201401+i)
for i in range(12):
   all_time_axis.append(201501+i)
for i in range(12):
   all_time_axis.append(201601+i)
for i in range(10):
   all_time_axis.append(201701+i)

# def get_wifi_dict(df):
