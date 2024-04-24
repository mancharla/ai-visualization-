import pandas as pd
f = pd.DataFrame({'Weather':['Sunny', 'Rainy', 'Sunny', 'Sunny'],
                'Wind':['Mild', 'Mild', 'High', 'Mild'],
                'Temp':['Moderate', 'Mild', 'Moderate', 'Mild'],
                'go':['Yes', 'No', 'Yes', 'Yes']})
print(f.columns)

from sklearn.naive_bayes import GaussianNB as g
from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split as tt

l = le()
for i in f.columns:
    f[i] = l.fit_transform(f[i])
    x = f.iloc[:, :3]
    y = f.iloc[:, 3]

xtr, xte, ytr, yte = tt(x, y, test_size=0.3)
gg = g()
gg.fit(xtr, ytr)
y_pred = gg.predict(xte)

from sklearn.metrics import accuracy_score
print(accuracy_score(yte, y_pred))
