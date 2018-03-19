import pandas as pd
file = open('../data/yancheng_train_20171226.csv')
data = pd.read_csv(file)  # 你的原始数据存放的路径
d=data[['sale_date','class_id','sale_quantity']]
data1=d.groupby([d['class_id'],d['sale_date']]).sum()
result = pd.DataFrame(data1)
result.to_csv('tidied.csv')