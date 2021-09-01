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

train_df = pd.read_csv("/home/mw/input/work8057/ѵ��Ӫ.csv")
test_df  = pd.read_csv("/home/mw/input/work8057/���Լ�.csv")
train_df.columns=['����', '�Ա�', '����', '����', '���', '����ָ��', '������Χ', '��Χ', '���Ѫѹ', '���Ѫѹ',
       '�õ��̴�', '�����̴�', '�ܵ��̴�', 'Ѫ֬�쳣', 'PVD', '�����', '����', 'δ��', '����', '������Դ',
       '��������', '����', '��Ѫѹ', '��ͥ��Ѫѹ', '����', '��������', '����', '�������', '����ƣ��',
       'ALF', 'ID']
test_df.columns=['����', '�Ա�', '����', '����', '���', '����ָ��', '������Χ', '��Χ', '���Ѫѹ', '���Ѫѹ',
       '�õ��̴�', '�����̴�', '�ܵ��̴�', 'Ѫ֬�쳣', 'PVD', '�����', '����', 'δ��', '����', '������Դ',
       '��������', '����', '��Ѫѹ', '��ͥ��Ѫѹ', '����', '��������', '����', '�������', '����ƣ��',
       'ID']

num_columns = ['����','����','���','����ָ��', '��Χ', '���Ѫѹ', '���Ѫѹ',
                '�õ��̴�', '�����̴�', '�ܵ��̴�','����']
zero_to_one_columns = ['������Χ','Ѫ֬�쳣','PVD']
str_columns = ['�Ա�','����','�����','����','δ��','������Դ','��������','����','��Ѫѹ',
                '��ͥ��Ѫѹ', '����', '��������','�������', '����ƣ��','ID']

# �ַ�����
for i in tqdm(str_columns):
    lbl = LabelEncoder()
    train_df[i] = lbl.fit_transform(train_df[i].astype(str))
    test_df[i] = lbl.fit_transform(test_df[i].astype(str))

# ��ֵ��һ��

train_df[num_columns] = MinMaxScaler().fit_transform(train_df[num_columns])
test_df[num_columns]  = MinMaxScaler().fit_transform(test_df[num_columns])

# ��ֵ���
train_df.fillna(0,inplace=True)
test_df.fillna(0,inplace=True)

all_columns = [i for i in train_df.columns if i not in ['����','ID']]
train_x,train_y = train_df[all_columns].values,train_df['����'].values
test_columns=['����', '�Ա�', '����', '����', '���', '����ָ��', '������Χ', '��Χ', '���Ѫѹ', '���Ѫѹ',
       '�õ��̴�', '�����̴�', '�ܵ��̴�', 'Ѫ֬�쳣', 'PVD', '�����', '����', 'δ��', '����', '������Դ',
       '��������', '����', '��Ѫѹ', '��ͥ��Ѫѹ', '����', '��������', '����', '�������', '����ƣ��',
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