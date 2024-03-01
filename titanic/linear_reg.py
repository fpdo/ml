#!/usr/bin/python3 

import glob
from titanic_data import Titanic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class LinearReg:
  def __init__(self, train_path, test_path):
    self.titanic = Titanic(train_path, test_path)

  def create_model(self, test_size):
    # What we want to predict
    y = self.titanic.train_data["Survived"]
    # Factors to account on the predictions
    x = self.titanic.train_data.drop("Survived", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=101)
    lin_reg = LogisticRegression(max_iter=500000)
    self.lin_reg_model = lin_reg.fit(x_train, y_train)
    return x_test, y_test

  def evaluate(self, dataset):
    return self.lin_reg_model.predict(dataset)


if __name__ == "__main__":
  model = LinearReg("train.csv", "test.csv")
  model.titanic.cleanup()

  # Train model, and return test data
  x_test, y_test = model.create_model(test_size=0.4)

  # Evaluate model
  prediction = model.evaluate(x_test)

  # Check model scores
  # | True positive - False positive |
  # | False negative - True negative |
  print(confusion_matrix(y_test, prediction))
  print(f"Model score: {model.lin_reg_model.score(x_test, y_test)}")

  # # Apply model to real data we want to predict
  model.titanic.append_data("test", "Survived", model.evaluate(model.titanic.test_data))
  model.titanic.append_data("test", "PassengerId", model.titanic.test_passenger_id)

  # Create submission file
  pattern = f"solutions/submission_ln*.csv"
  submission_data = model.titanic.test_data[["PassengerId", "Survived"]]

  files = sorted(glob.glob(pattern))
  last_submission = 0
  if len(files) > 0:
    last_submission = int(files[-1][-5]) 
  submission_data.to_csv(f"solutions/submission_ln{str(last_submission+1)}.csv", index=False)
