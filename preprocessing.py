from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(
    r'E:\Documents\College\VS Code\git\Tugas-Pertemuan-ke-3-Preprocessing-Data\Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("\n")
print("Menghilangkan Missing Value (nan)")
print(X)

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("\n")
print("Encoding data kategori (Atribut)")
print(X)

le = LabelEncoder()
Y = le.fit_transform(Y)
print("\n")
print("Encoding data kategori (Class / Label)")
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1)
print("\n")
print("Membagi dataset ke dalam training set dan test set")
print(X_train)
print("\n")
print(X_test)
print("\n")
print(Y_train)
print("\n")
print(Y_test)

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("\n")
print("Feature Scaling")
print(X_train)
print("\n")
print(X_test)
