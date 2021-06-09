import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv('Berlin-population-2020-10-17.csv')
df = df.dropna()

y  = df['Annual Change'].values
X  = df.iloc[:,:-1]
# Test Commit in dev branch
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


print('F1 Score: ', metrics.f1_score(y_test, y_pred, average='macro'))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred,average='macro'))



