import numpy as np
from sklearn.model_selection import KFold, train_test_split
from collections import Counter

data = np.load("../mel_data_nsynth/specs.npz", allow_pickle=True)

X = data["specs"]
Y = data["targets"]

Y_classes = np.argmax(Y[:, 0, :], axis=1)

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.2, random_state=42,  shuffle=True, stratify=Y_classes
)

Y_classes_t = np.argmax(Y_temp[:, 0, :], axis=1)

X_param, X_ridge, Y_param, Y_ridge = train_test_split(
    X_temp, Y_temp, test_size=1/2, random_state=42, shuffle=True, stratify=Y_classes_t
)

n_total = len(X)

print("Total:", n_total)
print("Train:", len(X_train), len(X_train)/n_total)
print("Param:", len(X_param), len(X_param)/n_total)
print("Ridge:", len(X_ridge), len(X_ridge)/n_total)


print("Train:", Counter(np.argmax(Y_train[:, 0, :], axis=1)))
print("Param:", Counter(np.argmax(Y_param[:, 0, :], axis=1)))
print("Ridge:", Counter(np.argmax(Y_ridge[:, 0, :], axis=1)))

np.savez("../mel_data_nsynth/mel_train.npz", specs=X_train, targets=Y_train)
np.savez("../mel_data_nsynth/mel_param.npz", specs=X_param, targets=Y_param)
np.savez("../mel_data_nsynth/mel_ridge.npz", specs=X_ridge, targets=Y_ridge)
