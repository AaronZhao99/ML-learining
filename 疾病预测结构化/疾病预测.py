import os
import pandas as pd
import warnings
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from catboost import CatBoostClassifier

train_df = pd.read_csv("/home/mw/input/work8057/训练营.csv")
test_df  = pd.read_csv("/home/mw/input/work8057/测试集.csv")
train_df.columns=['年龄', '性别', '区域', '体重', '身高', '体重指数', '肥胖腰围', '腰围', '最高血压', '最低血压',
       '好胆固醇', '坏胆固醇', '总胆固醇', '血脂异常', 'PVD', '体育活动', '教育', '未婚', '收入', '护理来源',
       '视力不佳', '饮酒', '高血压', '家庭高血压', '糖尿病', '家族糖尿病', '肝炎', '家族肝炎', '慢性疲劳',
       'ALF', 'ID']
test_df.columns=['年龄', '性别', '区域', '体重', '身高', '体重指数', '肥胖腰围', '腰围', '最高血压', '最低血压',
       '好胆固醇', '坏胆固醇', '总胆固醇', '血脂异常', 'PVD', '体育活动', '教育', '未婚', '收入', '护理来源',
       '视力不佳', '饮酒', '高血压', '家庭高血压', '糖尿病', '家族糖尿病', '肝炎', '家族肝炎', '慢性疲劳',
       'ID']

num_columns = ['年龄','体重','身高','体重指数', '腰围', '最高血压', '最低血压',
                '好胆固醇', '坏胆固醇', '总胆固醇','收入']
zero_to_one_columns = ['肥胖腰围','血脂异常','PVD']
str_columns = ['性别','区域','体育活动','教育','未婚','护理来源','视力不佳','饮酒','高血压',
                '家庭高血压', '糖尿病', '家族糖尿病','家族肝炎', '慢性疲劳','ID']

# 字符编码
for i in tqdm(str_columns):
    lbl = LabelEncoder()
    train_df[i] = lbl.fit_transform(train_df[i].astype(str))
    test_df[i] = lbl.fit_transform(test_df[i].astype(str))

# 数值归一化

train_df[num_columns] = MinMaxScaler().fit_transform(train_df[num_columns])
test_df[num_columns]  = MinMaxScaler().fit_transform(test_df[num_columns])

# 空值填充
train_df.fillna(0,inplace=True)
test_df.fillna(0,inplace=True)

all_columns = [i for i in train_df.columns if i not in ['肝炎','ID']]
train_x,train_y = train_df[all_columns].values,train_df['肝炎'].values
test_columns=['年龄', '性别', '区域', '体重', '身高', '体重指数', '肥胖腰围', '腰围', '最高血压', '最低血压',
       '好胆固醇', '坏胆固醇', '总胆固醇', '血脂异常', 'PVD', '体育活动', '教育', '未婚', '收入', '护理来源',
       '视力不佳', '饮酒', '高血压', '家庭高血压', '糖尿病', '家族糖尿病', '肝炎', '家族肝炎', '慢性疲劳',
       'ID']
test_x  = test_df[test_columns].values

submission=pd.DataFrame(columns=['id', 'ALF'])
submission['id']=test_df['ID']
submission['ALF'] = 0


kfold = StratifiedKFold(n_splits=5, shuffle=False)
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    loss_function='Logloss'
    )
for train, valid in kfold.split(train_x, train_y):
    X_train, Y_train = train_x[train], train_y[train]
    X_valid, Y_valid = train_x[valid], train_y[valid]
    model.fit(X_train,Y_train, eval_set=(X_valid, Y_valid),use_best_model=True)
    Y_valid_pred_prob = model.predict_proba(X_valid)
    submission['ALF'] += model.predict_proba(test_x)[:,1] / 5

submission.to_csv('submission.csv',index=False)