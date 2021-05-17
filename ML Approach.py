import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn import metrics

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

kepler = pd.read_csv('/Users/mertefesevim/Documents/GitHub/Exoplanet-Hunting/NASA_dataset.csv')

kepler = kepler.drop(['rowid','kepid','kepoi_name','kepler_name','koi_pdisposition','koi_score'], axis=1)

#print(kepler.notnull())

cols = ['koi_tce_delivname', 'koi_disposition']
kepler[cols] = kepler[cols].apply(lambda x: pd.factorize(x)[0] + 1)

for column in kepler.columns[kepler.isna().sum() > 0]:
    kepler[column] = kepler[column].fillna(kepler[column].mean())

kepler_target = kepler['koi_disposition']
kepler=kepler.drop(['koi_disposition'], axis=1)

#training

X_train, X_test, y_train, y_test = train_test_split(kepler, kepler_target, train_size=0.8, test_size=0.2, random_state=0)

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: {}.".format(accuracy_score(y_test, y_pred)))
print("Precision: {}.".format(precision_score(y_test, y_pred)))
print("Recall: {}".format(recall_score(y_test, y_pred)))
print("F1_Score: {}".format(f1_score(y_test, y_pred)))

