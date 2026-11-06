import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# 讀取資料集
path = 'Campus Recruitment/Placement_Data_Full_Class.csv'
CR_dataset = pd.read_csv(path)


#######################################################################################################################################


# 屬性分析


# # 檢查缺失值
# missing_values = CR_dataset.isnull().sum()
# print('missing value:', missing_values[missing_values > 0], '\n')
CR_dataset = CR_dataset.drop(columns=['sl_no', 'salary']) # 去除 sl_no 與 salary 屬性
# print(CR_dataset)


# # 檢查各數值屬性統計數據
# CR_dataset_statistic = CR_dataset.describe()
# print(CR_dataset_statistic)


#####################################################################################################################################


# 前處理


# 歸一化
columns_to_normalize = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
scaler = MinMaxScaler()
CR_dataset[columns_to_normalize] = scaler.fit_transform(CR_dataset[columns_to_normalize])
# print(CR_dataset[columns_to_normalize].describe())


# 二值化(one hot encoding)
columns_to_encode = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = encoder.fit_transform(CR_dataset[columns_to_encode])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))
CR_dataset_ohe = pd.concat([CR_dataset, encoded_df], axis=1)
CR_dataset_ohe = CR_dataset_ohe.drop(columns_to_encode, axis=1)
# print(CR_dataset_ohe.describe())


# 取樣
X = CR_dataset_ohe.drop(columns=['status_Placed']) 
y = CR_dataset_ohe['status_Placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


#################################################################################################################################


# 套用KNN分類器
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')


# 計算所有屬性的accuracy
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_all_feature = accuracy_score(y_test, y_pred)
print('accuracy_all_feature:', accuracy_all_feature, '\n')


# 計算單一屬性的accuracy
accuracy_each_feature = pd.Series(index=X.columns)
for feature in accuracy_each_feature.index:
    knn.fit(X_train[[feature]], y_train)
    y_pred = knn.predict(X_test[[feature]])
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_each_feature[feature] = accuracy
print('accuracy_each_feature:\n', accuracy_each_feature.sort_values(ascending=False), '\n')


# 選擇 hsc_p 屬性( accuracy 最高)，並觀察加入其他屬性後的變化
feature_select = ['hsc_p'] # 嘗試加入其他屬性，但 accuracy 反而下降
X_feature_select = CR_dataset_ohe[feature_select]
y = CR_dataset_ohe['status_Placed']
X_train_feature_select, X_test_feature_select, y_train, y_test = train_test_split(X_feature_select, y, train_size=0.8, random_state=0)
knn.fit(X_train_feature_select, y_train)
y_pred = knn.predict(X_test_feature_select)
# print(X_train_feature_select)


#######################################################################################################################################


# 評估效能


# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('conf_matrix:\n', conf_matrix, '\n')


# accuracy
accuracy = accuracy_score(y_test, y_pred)
print('accuracy:', accuracy, '\n')


# error rate
error_rate = 1 - accuracy
print('error_rate:', error_rate, '\n')


# precision
precision = precision_score(y_test, y_pred)
print('precision:', precision, '\n')


# recall
recall = recall_score(y_test, y_pred)
print('recall:', recall, '\n')


# f-measure
f_measure = f1_score(y_test, y_pred)
print('f-measure:', f_measure, '\n')


#######################################################################################################################################


# 屬性重要程度


feature_importance = pd.Series(index=X.columns)
accuracy_min = accuracy_each_feature.min()

for feature in feature_importance.index:
    feature_importance[feature] = accuracy_each_feature[feature] - accuracy_min

feature_importance_sum = feature_importance.sum()

for feature in feature_importance.index:
    feature_importance[feature] = feature_importance[feature] / feature_importance_sum

print('feature_importance:\n', feature_importance.sort_values(ascending=False), '\n')