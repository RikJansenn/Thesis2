import numpy as np
from sklearn.model_selection import KFold, train_test_split

data = np.load("specs.npz", allow_pickle=True)

X = data["specs"]
Y = data["targets"]

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.2, random_state=42,  shuffle=True
)

X_param, X_ridge, Y_param, Y_ridge = train_test_split(
    X_temp, Y_temp, test_size=1/2, random_state=42, shuffle=True
)

n_total = len(X)

print("Total:", n_total)
print("Train:", len(X_train), len(X_train)/n_total)
print("Param:", len(X_param), len(X_param)/n_total)
print("Ridge:", len(X_ridge), len(X_ridge)/n_total)

np.savez("mel_train.npz", specs=X_train, targets=Y_train)
np.savez("mel_param.npz", specs=X_param, targets=Y_param)
np.savez("mel_ridge.npz", specs=X_ridge, targets=Y_ridge)
