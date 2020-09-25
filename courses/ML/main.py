from sklearn.datasets import load_svmlight_file
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

def get_data(file):
    data = load_svmlight_file(file)
    return data[0], data[1]
X, y = get_data("a5a")
length = len(y)
kf = KFold(n_splits=3)
Cs = [0.01, 0.1, 1, 10, 100]
for C in Cs:
    svm = LinearSVC(C=C)
    acc = list()
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svm.fit(X_train, y_train)
        acc.append(svm.score(X, y))
    mean_acc = sum(acc) / len(acc)
    print("C:{}, acc:{}".format(C, mean_acc))
    acc = list()
svm = LinearSVC(C=0.1)
svm.fit(X, y)
X, y = get_data("a5a.t")
print(svm.score(X, y))