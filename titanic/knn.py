#!/usr/bin/python3 

import glob
import pandas as pd
import numpy as np
from titanic_data import Titanic
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

class Knn:
  def __init__(self, train_path, test_path):
    self.titanic = Titanic(train_path, test_path)

  def _select_k(self, x_train, y_train, x_test, y_test, boundary=40):
    error_rate = []
    for i in range(1, boundary):
      knn = KNeighborsClassifier(n_neighbors=i)
      knn.fit(x_train, y_train)
      pred = knn.predict(x_test)
      error_rate.append(np.mean(pred != y_test))

    return error_rate.index(min(error_rate)) + 1

  def create_model(self, test_size):
    scaler = StandardScaler()
    scaler.fit(self.titanic.train_data.drop("Survived", axis=1))
    scaler.fit(self.titanic.test_data)
    scld_train = scaler.transform(self.titanic.train_data.drop("Survived", axis=1))
    scld_test = scaler.transform(self.titanic.test_data)
    scaled_train = pd.DataFrame(scld_train, columns=self.titanic.train_data.columns[1:])
    scaled_test = pd.DataFrame(scld_test, columns=self.titanic.test_data.columns)

    y = self.titanic.train_data["Survived"]
    x = scaled_train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    n_neighbors = self._select_k(x_train, y_train, x_test, y_test)
    self.knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    self.knn_model.fit(x_train, y_train)
    return x_test, y_test, scaled_test
  
  def evaluate(self, dataset):
    return self.knn_model.predict(dataset)

if __name__ == "__main__":
  model = Knn("train.csv", "test.csv")
  model.titanic.cleanup()

  x_test, y_test, scaled_test_data = model.create_model(test_size=0.4)

  prediction = model.evaluate(x_test)

  # Check model scores
  # | True positive - False positive |
  # | False negative - True negative |
  print(confusion_matrix(y_test, prediction))
  print(f"Model score: {model.knn_model.score(x_test, y_test)}")

  # # Apply model to real data we want to predict
  model.titanic.append_data("test", "Survived", model.evaluate(scaled_test_data))
  model.titanic.append_data("test", "PassengerId", model.titanic.test_passenger_id)

  # Create submission file
  pattern = f"solutions/knn*.csv"
  submission_data = model.titanic.test_data[["PassengerId", "Survived"]]

  files = sorted(glob.glob(pattern))
  last_submission = 0
  if len(files) > 0:
    last_submission = int(files[-1][-5]) 
  submission_data.to_csv(f"solutions/knn{str(last_submission+1)}.csv", index=False)