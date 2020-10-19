import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# reading file
input_file = "train-dup.csv"
df = pd.read_csv(input_file, header = 0)
original_headers = list(df.columns.values)
# loading to matrix
X = df.iloc[:,:-1].values
y = df.iloc[:,-1:].values
# convert y from text to number
le = preprocessing.LabelEncoder()
le.fit(["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"])
print(list(le.classes_))
transformed_y = le.transform(y)
# using KFold to test LinearRegression
kf = KFold(n_splits=5)
print(kf.get_n_splits(X))
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index.size, "TEST:", test_index.size)
    print("TRAIN:", train_index[0], "TEST:", test_index[0])
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = transformed_y[train_index], transformed_y[test_index]
    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_test, y_test))
# using built-in cross validation to test LinearRegression
scoresLR = cross_val_score(LinearRegression(),X,transformed_y, cv=5)
print("score Linear Regression:")
print(scoresLR)
# train with SVM with random train_test_split
print("### SVM")
X_train, X_test, y_train, y_test = train_test_split(X,transformed_y, test_size=0.2, random_state=42)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
print("predict:",clf.predict([X_test[0]]))
print(y_test[0])
print(clf.predict([X_test[1]]))
print(y_test[1])
# train with SVM with random cross_val_score
scores = cross_val_score(clf, X, transformed_y, cv=5)
print("score SVM:")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# train with k-nearest neighbour
print("### k-nearest neighbour")
neigh = KNeighborsClassifier(n_neighbors=3)
# using preexisting train test split:
neigh.fit(X_train, y_train)
print("predict: ",neigh.predict([X_test[2]]))
print(y_test[2])
# test with cross_val_score
scores = cross_val_score(neigh, X, transformed_y, cv=5)
print(scores)