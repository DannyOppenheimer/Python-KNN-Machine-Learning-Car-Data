# K Nearest Neighbors Machine Learning using Tensorflow Libraries
# ---Credit---
# Marko Bohanec. UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/car+evaluation].

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

# read in our car.data file
carData = pd.read_csv("car.data", sep=",")

# transforming values into numerical values
# if there was an array like [low, medium, high] then the encoder would change it to [0, 1, 2]
encoder = preprocessing.LabelEncoder()
buying = encoder.fit_transform(list(carData["buying"]))
maint = encoder.fit_transform(list(carData["technical_charac"]))
doors = encoder.fit_transform(list(carData["doors"]))
persons = encoder.fit_transform(list(carData["persons"]))
lug_boot = encoder.fit_transform(list(carData["lug_boot"]))
safety = encoder.fit_transform(list(carData["safety"]))
classc = encoder.fit_transform(list(carData["class"]))

# split data into a list of goals and a list of training data
X = list(zip(buying, maint, doors, persons, lug_boot, safety))
Y = list(classc)

# x_train and y_train is training data the computer can read and learns off of
# x_test and y_test is the prediction data the computer is given to calculate new final grade
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# how many closest points it should look for
k_value = KNeighborsClassifier(n_neighbors=9)

# slap the data into the line
k_value.fit(x_train, y_train)
# accuracy score of the data
accuracy = k_value.score(x_test, y_test)
print("Accuracy: ", "%.01f" % (accuracy * 100), "%", sep="")

# predict for the info on the data collected
predicted_val = k_value.predict(x_test)
# replacement names for the 1-4 values
rep_names = ["bad", "okay", "good", "very good"]

# run through all our testing data
for x in range(len(predicted_val)):
    print("Predicted: ", rep_names[predicted_val[x]], " | Data: ", x_test[x], " | Actual: ", rep_names[y_test[x]], sep="")
    # not needed right now, but shows the distances of all the neighbors
    neighbor_distance = k_value.kneighbors([x_test[x]], 9, True)
