import pandas as pd
import numpy as np
import GPUtil

from sklearn.model_selection import train_test_split

marking = pd.read_csv('C:\\Users\\Eva.Locusteanu\\PycharmProjects\\detr\\models\\train.csv')

y = marking.pop('source')
X = marking

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.005, random_state=42, stratify=y)

print(len(marking))



print("Entire set shape is:", marking.shape)
print("X train shape:", X_train.shape)
print("y train shape:", y_train.shape)

print("X test shape:", X_test.shape)
print("y test shape:", y_test.shape)

print(y_test.head())
print(type(y_test))
print(X_test.head())

#mini = pd.merge(left = X_test, right = y_test)
                #left_on = 'image_id', right_on = 'image_id'

X_test['source'] = y_test

new = X_test

print("New:", new.shape) #15K images
print(new.head())

new.to_csv("Point_five_Percent_MiniTrainingData.csv")

