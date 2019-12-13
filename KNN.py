# Marko Bohanec. UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/car+evaluation].

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

carData = pd.read_csv("car.data", sep=",")

# transforming values into numerical values
encoder = preprocessing.LabelEncoder()
buying = encoder.fit_transform(list(carData["buying"]))
maint = encoder.fit_transform(list(carData["maint"]))
doors = encoder.fit_transform(list(carData["doors"]))
persons = encoder.fit_transform(list(carData["persons"]))
lug_boot = encoder.fit_transform(list(carData["lug_boot"]))
safety = encoder.fit_transform(list(carData["safety"]))
classc = encoder.fit_transform(list(carData["class"]))

# what computer is attempting to predict
predictGoal = "class"

# split data into a list of goals and a list of training data
X = list(zip(buying, maint, doors, persons, lug_boot, safety))
Y = list(classc)

# x_train and y_train is training data the computer can read and learns off of
# x_test and y_test is the prediction data the computer is given to calculate new final grade
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# how many closest points it should look for
kValue = KNeighborsClassifier(n_neighbors=9)

kValue.fit(x_train, y_train)
accuracy = kValue.score(x_test, y_test)
print("Accuracy: ", "%.01f" % (accuracy * 100), "%", sep="")

predictedVal = kValue.predict(x_test)
names = ["bad", "okay", "good", "very_good"]

for x in range(len(predictedVal)):
    print("Predicted: ", names[predictedVal[x]], " | Data: ", x_test[x], " | Actual: ", names[y_test[x]], sep="")
