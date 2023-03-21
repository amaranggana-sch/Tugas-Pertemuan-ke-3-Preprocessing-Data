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
