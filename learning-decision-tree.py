## First thing first, import several required libraries to read file, to do plot, and to do numerical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

iris = pd.read_csv("C:/Users/qalha/(1) Learning Python/datasets/Iris.csv") ## This function is used to open/read the csv data from local database
iris.head() ## this function is used to see first 5 data in dataset

iris.drop('Id', axis=1, inplace=True) ## this function is used to drop/delete the unuse data, as the parameter DataFrame(data-name, axis=1, inplace=True), axis=1 means that for row, and implace=True means that the drop function will affect the whole data
iris.head()

## Next we import several libraries from scikit-learn to do split a dataset, decision tree algorithm, and to see the score/accuracy of the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

train, test = train_test_split(iris, test_size = 0.3, random_state = 2) ## this function is used to split a dataset into test-data (20%) and train-data (30%) with random_state=2
print(train.shape)
print(test.shape)

## Now we divide test and train variable into train-data and test-data for x and y separately. x is input and y is output
train_x = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
train_y = train.Species
test_x = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y = test.Species

## Now we train a model using decision tree algorithm
model = DecisionTreeClassifier()
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print('The accuracy of the Decision Tree is: %.2f' %metrics.accuracy_score(prediction,test_y)) ## this function is to see the accuracy of our trained model

## Now I want to see how the decision tree works by using graphviz visualization
from sklearn.tree import export_graphviz

export_graphviz(model, out_file='tree.dot', class_names=['iris-setosa', 'iris-versicolor', 'iris-virginica'],
              feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], impurity=False, filled=True)

import pydot
graphs = pydot.graph_from_dot_file("tree.dot")
graph = graphs[0]
graph.write_png("output.png")

from sklearn.model_selection import cross_val_score

iris2 = np.array(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
scores = cross_val_score(model, iris2, iris.Species, cv=10)
print('Cross-validation score: {}'.format(scores))

print('Average cross-validation score: {:.2f}'.format(scores.mean()))
