#Applying random forest walk to clinical heart data to predict death rate
#Author: Nicholas Mosca

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


# importing data
file = "heart_failure_clinical_records_dataset.csv"
#All Data
total_data = pd.read_csv(file,na_values=" NaN")
total_data =total_data.dropna()

data_feature_names = list(total_data.columns)
data_summary = total_data.describe() # statistic description 

#cleaning data
Trimmed_data = total_data.drop('DEATH_EVENT',axis = 1)
Trimmed_data = Trimmed_data.drop('time',axis = 1)  # not a clinical value

Target_data = np.array(total_data['DEATH_EVENT']) # binary data

#Feature Selection / ranking via Random Forest and gini importance
X = Trimmed_data
y = Target_data
# X = X[~np.isnan(X)]
# y = y[~np.isnan(y)]

#setting up training and testing sets for 30%test , 70% training (Paper guidelines)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# random forest classifier with 10k decision trees
#the dataset via bootstrap 
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1) 
# Training the classifier

clf.fit(X_train, y_train)

#gini importance of each feature
gini_importance_results = []
for feature in zip(data_feature_names, clf.feature_importances_):
    gini_importance_results.append(feature)

#converting data to dict
gini_importance_results = dict(gini_importance_results)

#sorting data
gini_importance_results = {k: v for k, v in sorted(gini_importance_results.items(),reverse=True, key=lambda item: item[1])}
gini_features = list(gini_importance_results.keys())
gini_values = list(gini_importance_results.values())


#Permutation Based Feature Importance Results

perm_importance = permutation_importance(clf,X_test,y_test)
sorted_idx = perm_importance.importances_mean.argsort()

perm_importance_data = perm_importance.importances_mean[sorted_idx]
perm_importance_data = np.sort(perm_importance_data)



#Figure generation

fig = plt.figure(figsize=(30,20),dpi= 800)
ax1 = fig.add_subplot(221)
ax1.barh(gini_features[::-1], perm_importance_data)
ax1.set_xlabel("Permutation Importance")
ax1.set_title('Error rate per feature via Permutation Importance')


#plotting for fig B
plt.rcdefaults()

ax2 = fig.add_subplot(222)
y_pos = np.arange(len(gini_features))

ax2.barh(y_pos, gini_values, align='center')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(gini_features)
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.set_xlabel('Gini Impurity')
ax2.set_title('Feature Ranking via Gini impurity')

