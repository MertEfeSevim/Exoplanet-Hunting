import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm

kepler = pd.read_csv('/Users/mertefesevim/Documents/GitHub/Exoplanet-Hunting/NASA_dataset.csv')

kepler = kepler.drop(columns=['rowid',
                              'kepid',
                              'kepoi_name',
                              'kepler_name',
                              'koi_pdisposition',
                              'koi_score',
                              'koi_teq_err1',
                              'koi_teq_err2'], axis=1)

cols = ['koi_tce_delivname', 'koi_disposition']
kepler[cols] = kepler[cols].apply(lambda x: pd.factorize(x)[0] + 1)

for column in kepler.columns[kepler.isna().sum() > 0]:
    kepler[column] = kepler[column].fillna(kepler[column].mean())

kepler_target = kepler['koi_disposition']
kepler = kepler.drop(['koi_disposition'], axis=1)

#print(kepler.shape)
#print(np.any(np.isnan(kepler)))
#print(np.all(np.isfinite(kepler)))
kepler[np.isfinite(kepler) == True] = 0


# SVM Model Training
print("SVM Model Training")
X_train, X_test, y_train, y_test = train_test_split(kepler, kepler_target, test_size=0.25)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Precision Score : ",precision_score(y_test, y_pred, average='micro'))
print("Recall Score : ",recall_score(y_test, y_pred, average='micro'))
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print("F1 Score : ",f1_score(y_test, y_pred, average='micro'))

print("------- -------- ------")

# Decision Tree Model Training
print("Decision Tree Model Training")

X_train, X_test, y_train, y_test = train_test_split(kepler, kepler_target, test_size=0.3)


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Precision Score : ",precision_score(y_test, y_pred, average='micro'))
print("Recall Score : ",recall_score(y_test, y_pred, average='micro'))
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print("F1 Score : ",f1_score(y_test, y_pred, average='micro'))

'''
print("------- -------- ------")

# Random Forest Model Training
print("Random Forest Model Training")
X_train, X_test, y_train, y_test = train_test_split(kepler, kepler_target, test_size=0.25)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X_train, y_train = make_classification(n_samples=1000, n_features=4, shuffle=False)
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf = clf.fit(X_train, y_train)

print("Precision Score : ",precision_score(y_test, y_pred, average='micro'))
print("Recall Score : ",recall_score(y_test, y_pred, average='micro'))
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print("F1 Score : ",f1_score(y_test, y_pred, average='micro'))
'''