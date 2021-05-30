import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

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

X_train, X_test, y_train, y_test = train_test_split(kepler, kepler_target, test_size=0.2)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 score:", metrics.f1_score(y_test, y_pred))
