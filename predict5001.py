# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:12:06 2019

@author: 范嘉昊
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingRegressor
import warnings
warnings.filterwarnings("ignore")


def purchaseDate_process(dateStr):
    if str(dateStr) != 'nan':
        date_list = dateStr.split(' ')
        month = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,\
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,}
        dateList = [int(date_list[2]), month[date_list[0]], int(date_list[1][:-1])]
        return dateList
    else:
        return [dateStr, dateStr, dateStr]

def releaseDate_process(dateStr):
    if str(dateStr) != 'nan':
        if dateStr != 'Nov 10, 2016' and len(dateStr) >= 11:
            date_list = dateStr.split(' ')
            month = {'Jan,': 1, 'Feb,': 2, 'Mar,': 3, 'Apr,': 4, 'May,': 5, 'Jun,': 6,\
                     'Jul,': 7, 'Aug,': 8, 'Sep,': 9, 'Oct,': 10, 'Nov,': 11, 'Dec,': 12,}
            dateList = [int(date_list[2]), month[date_list[1]], int(date_list[0])]
            return dateList
        elif dateStr != 'Nov 10, 2016' and len(dateStr) < 11:
            date_list = dateStr.split('-')
            month = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,\
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,}
            dateList = [int(date_list[2])+2000, month[date_list[1]], int(date_list[0])]
            return dateList
        else:
            return [2016, 11, 10]
    else:
        return [dateStr, dateStr, dateStr]

def purchaseToday(dateStr):
    if str(dateStr) != 'nan':
        dateList = purchaseDate_process(dateStr)
        before = datetime.date(dateList[0], dateList[1], dateList[2])
        current = datetime.date(2019, 10, 1)
        number_of_days = current.__sub__(before).days
        return number_of_days
    else:
        return dateStr
    
def releaseToday(dateStr):
    if str(dateStr) != 'nan':
        dateList = releaseDate_process(dateStr)
        before = datetime.date(dateList[0], dateList[1], dateList[2])
        current = datetime.date(2019, 10, 1)
        number_of_days = current.__sub__(before).days
        return number_of_days
    else:
        return dateStr


original = pd.read_csv('DataTable/train.csv')
wait_for_predict = pd.read_csv('DataTable/test.csv')
dummies = pd.read_csv('DataTable/dummy.csv').iloc[:,1:]

original_len = len(original.iloc[:,:].values)
wait_for_predict_len = len(wait_for_predict.iloc[:,:].values)

outputID = wait_for_predict['id']
original = original[['is_free', 'price',\
                     'purchase_date',  'release_date', 'total_positive_reviews',\
                     'total_negative_reviews', 'playtime_forever']]
wait_for_predict = wait_for_predict[['is_free', 'price',\
                     'purchase_date',  'release_date', 'total_positive_reviews',\
                     'total_negative_reviews']]
wait_for_predict['playtime_forever'] = [0] * wait_for_predict_len

#将train.csv和test.csv合并成为一个新表，便于填补缺失数据
dataset = pd.concat([original, wait_for_predict], axis = 0)    
dataset = dataset.reset_index(drop=True)
result = dataset['playtime_forever']
dataset.drop(labels=['playtime_forever'], axis=1,inplace = True)

#将genres、categories和tags改分为多个维度
"""
genres_dummies = dataset['genres'].str.get_dummies(",") 
genres_dummies.columns = ['genres_action', 'genres_adventure', 'genres_a&m',\
                          'genres_AudioPro', 'genres_casual', 'genres_d&i',\
                          'genres_earlyAccess', 'genres_freePlay', 'genres_gore',\
                          'genres_indie', 'genres_massMultiplayer', 'genres_nudity',\
                          'genres_rpg', 'genres_racing', 'genres_sexualContent',\
                          'genres_simulation', 'genres_sports', 'genres_strategy',\
                          'genres_utilities', 'genres_violent']
dataset = pd.concat([dataset, genres_dummies], axis = 1) 

category_dummies = dataset['categories'].str.get_dummies(",") 
category_dummies.columns = ['category_captionAva', 'category_co-op', 'category_commentaryAva',\
                             'category_c-pM', 'category_fcs', 'category_i-aP', 'category_iss',\
                             'category_ile', 'category_lco-op', 'category_lmp',\
                             'category_mmo', 'category_m-p', 'category_oco-op',\
                             'category_omp', 'category_pcs', 'category_rpop', 'category_rpotv',\
                             'category_rpotablet', 'category_s/ss', 'category_sp',\
                             'category_stats', 'category_sa', 'category_steamcloud',\
                             'category_steamL', 'category_steamTC', 'category_steamWorkshop',\
                             'category_steamVR', 'category_VRSupport', 'category_vace']
dataset = pd.concat([dataset, category_dummies], axis = 1)

tag_dummies = dataset['tags'].str.get_dummies(",") 
dataset = pd.concat([dataset, tag_dummies], axis = 1)
"""
dataset['is_free'] = dataset['is_free'].apply(lambda x: 1 if x==True else 0)
#dataset.drop(labels=['genres', 'categories', 'tags'], axis = 1, inplace = True)

#对表中的时间列进行处理，改为距离现在多久
dataset['purchase_date_len'] = dataset['purchase_date'].apply(lambda x: purchaseToday(x)) 
dataset['release_date_len'] = dataset['release_date'].apply(lambda x: releaseToday(x))

dataset['purchase_date_year'] = dataset['purchase_date'].apply(lambda x: 2019-purchaseDate_process(x)[0])
dataset['release_date_year'] = dataset['release_date'].apply(lambda x: 2019-releaseDate_process(x)[0])
dataset['purchase_date_month'] = dataset['purchase_date'].apply(lambda x: purchaseDate_process(x)[1])
#dataset['release_date_month'] = dataset['release_date'].apply(lambda x: releaseDate_process(x)[1])
#dataset['puechase_date_week'] = dataset['purchase_date_len']
#dataset['puechase_date_week'] = dataset['puechase_date_week'].apply(lambda x: 2-(x%7) if 2-(x%7)>0 else 9-(x%7))

#dataset['purchase_To_release'] = dataset['release_date_len']-dataset['purchase_date_len']

#dataset['positive_review'] = dataset['total_positive_reviews'] + dataset['total_negative_reviews']
#dataset['positive_review'] = dataset['positive_review'].apply(lambda x: 1 if str(x) in [0,'nan'] else x)
#dataset['positive_review'] = dataset['total_positive_reviews']/dataset['positive_review']
dataset['hotPoint'] = dataset['total_positive_reviews'] + dataset['total_negative_reviews']

dataset.drop(labels=['purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews',\
                     'release_date_len'], axis=1,inplace = True)

dataset = pd.concat([dataset, dummies.iloc[:,[2]]], axis=1)

dataset = pd.concat([dataset, result], axis = 1)

play_or_not = original['playtime_forever'].apply(lambda x: 0 if x<=0.05 else 1)
y_pon = play_or_not.iloc[:].values



X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#----------------------------------缺失数据填补---------------------------------
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,:])
X[:,:] = imputer.transform(X[:,:])
X_positive = X[:, [-2]]
#------------------------------------------------------------------------------

from minepy import MINE
m = MINE()
save_columns = []
for i in range(0, len(X[0])):
    m.compute_score(X[:,i], y)
    #print(i, m.mic())
    if m.mic()>=0.1:
        save_columns.append(i)

X = X[:, save_columns]
df = pd.DataFrame(X)
from sklearn.feature_selection import VarianceThreshold
val_selection = VarianceThreshold(threshold=(0.1*(1-0.1)))
X = val_selection.fit_transform(X)
X = np.hstack((X_positive, X))

#------------------------------------------------------------------------------

#-----------------------------------切割数据集----------------------------------
X_old = X[:original_len, :]
X_old = np.delete(X_old, [220,312], axis=0)
y_old = y[:original_len]
y_old = np.delete(y_old, [220,312], axis=0)
X_new = X[original_len:, :]
#------------------------------------------------------------------------------

#------------------------------------特征缩放-----------------------------------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_old = sc_X.fit_transform(X_old)
X_new = sc_X.transform(X_new)
#sc_y = StandardScaler()
#y_old = sc_y.fit_transform(y_old)
#------------------------------------------------------------------------------



"""是否玩游戏？分类"""
from sklearn.svm import SVC
svc_pon = SVC(kernel='rbf', random_state=0, C=1250, gamma=0.003)
y_pon = np.delete(y_pon, [220,312], axis=0)
svc_pon.fit(X_old, y_pon)
new_pon_pred = svc_pon.predict(X_new)
X_old = np.column_stack((X_old, y_pon))
X_new = np.column_stack((X_new, new_pon_pred))

"""
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,5,10], 'kernel':['linear']},\
               {'C':[1200,1225,1250,1275,1300],'kernel':['rbf'],'gamma':[0.004,0.005,0.003,0.002]}]
grid_search = GridSearchCV(estimator=svc_pon, param_grid=parameters, scoring='accuracy',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_pon)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""

#-------------------------------划分训练集和测试集-------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#-----------------------------------模型制作----------------------------------
"""此处选择lightGBM模型进行拟合"""

import lightgbm as lgb
gbm = lgb.LGBMRegressor(learning_rate=0.01, n_estimators=5000, max_depth=6,\
                        num_leaves=12, min_data_in_leaf=18,\
                        min_sum_hessian_in_leaf=0.05, bagging_fraction=0.01,\
                        feature_fraction=0.75, reg_alpha=0.8, reg_lambda=0.65)
#gbm.fit(X_old, y_old)

from sklearn.svm import SVR
svr = SVR(C=98, gamma=0.9, kernel='rbf')
svr.fit(X_old, y_old)


from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_old, y_old)


from sklearn.linear_model import Lasso, Ridge
lasso = Lasso(alpha=0.0001)
lasso.fit(X_old, y_old)
ridge = Ridge(alpha=0.0001)
ridge.fit(X_old, y_old)

from sklearn.linear_model import BayesianRidge
bys = BayesianRidge()
bys.fit(X_old, y_old)

import xgboost
xgb = xgboost.XGBRegressor(n_estimators=5000, learning_rate=0.01, max_depth=4,\
                           min_child_weight=19, gamma=0.83, subsample=0.95,\
                           colsample_bytree=0.75, reg_alpha=0.51, reg_lambda=0.2)
xgb.fit(X_old, y_old)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=5000, random_state=0,\
                           max_depth=9, max_features='sqrt', min_samples_leaf=2,\
                           min_samples_split=2)
rf.fit(X_old, y_old)


models = [svr,rf,gbm,xgb]
sclf = StackingRegressor(regressors=models, meta_regressor=bys)
sclf.fit(X_old, y_old)

#------------------------------------------------------------------------------

#------------------------------------模型调参-----------------------------------
from sklearn.model_selection import GridSearchCV
"""
parameters = [{'num_leaves':range(2,40), 'max_depth':range(2,40)}]
grid_search = GridSearchCV(estimator=gbm, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'min_data_in_leaf':range(2,50), 'min_sum_hessian_in_leaf':[0.005]}]
grid_search = GridSearchCV(estimator=gbm, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'feature_fraction':[i/1000.0 for i in range(730,780,1)], 'bagging_fraction':[i/1000.0 for i in range(10,50,1)]}]
grid_search = GridSearchCV(estimator=gbm, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'reg_alpha':[i/100.0 for i in range(75,85,1)], 'reg_lambda':[i/100.0 for i in range(60,70,1)]}]
grid_search = GridSearchCV(estimator=gbm, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'learning_rate':[0.1,0.01,0.05], 'n_estimators':[50,100,150,200,500,1000]}]
grid_search = GridSearchCV(estimator=gbm, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""

"""
parameters = [{'n_estimators':[750,800,850,900,950,1000]}]
grid_search = GridSearchCV(estimator=xgb, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'max_depth': range(0,20), 'min_child_weight': range(0,20)}]
grid_search = GridSearchCV(estimator=xgb, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'gamma': [i/100.0 for i in range(10,100)]}]
grid_search = GridSearchCV(estimator=xgb, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'subsample': [i/100.0 for i in range(90,100,1)], 'colsample_bytree': [i/100.0 for i in range(70,80,1)]}]
grid_search = GridSearchCV(estimator=xgb, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'reg_alpha': [i/100.0 for i in range(50,60,1)], 'reg_lambda': [i/100.0 for i in range(20,30,1)]}]
grid_search = GridSearchCV(estimator=xgb, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'min_samples_leaf':range(2,20),\
               'max_features':['sqrt','log2'],'max_depth': range(2,20),'min_samples_split': range(2,20)}]
grid_search = GridSearchCV(estimator=rf, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
"""
parameters = [{'C':[1,5,10], 'kernel':['linear']},\
               {'C':range(80,100,1),'kernel':['rbf'],'gamma':[i/100.0 for i in range(80,100,1)]}]
grid_search = GridSearchCV(estimator=svr, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_old, y_old)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_
"""
#------------------------------------------------------------------------------

#-----------------------------Cross Validation---------------------------------
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = sclf, X=X_old, y=y_old, cv=10, scoring='neg_mean_squared_error')
rmse = np.sqrt(-accuracies).mean()

#------------------------------------------------------------------------------
output = pd.DataFrame(sclf.predict(X_new))
output.columns = ['playtime_forever']
output.insert(0, 'id', range(len(output)))
output['playtime_forever'] = output['playtime_forever'].apply(lambda x: 0 if float(x)<0 else x)
output.to_csv('prediction.csv', index = False)
