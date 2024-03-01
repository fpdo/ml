#!/usr/bin/python3 

import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class LinearReg:
  def __init__(self, train_path, test_path):
    self.train_data = pd.read_csv(train_path)
    self.test_data = pd.read_csv(test_path)

  def calculate_mean_by_class(self, cols_val, col_nam, model):
    dataset = self.train_data
    if model != "train":
      dataset = self.test_data

    arg1 = cols_val.iloc[0]
    pclass = cols_val.iloc[1]
    if pd.isnull(arg1):
      return dataset[dataset["Pclass"] == pclass][col_nam].mean()
    if arg1 == "LINE":
      return dataset[(dataset["Pclass"] == pclass) & (dataset["Ticket"] != "LINE")][col_nam].mean()
    return arg1

  def convert_categorical_column(self, col_nam, model):
    dataset = self.train_data
    if model != "train":
      dataset = self.test_data

    dummy = pd.get_dummies(dataset[col_nam], drop_first=True).astype(int)
    return pd.concat([dataset, dummy], axis=1)
  
  def format_ticker_value(self, ticket):
    ticket = ticket.split(" ")
    try:
      return float(ticket[0]) if len(ticket) == 1 else float(ticket[-1])
    except ValueError:
      return ticket[0] if len(ticket) == 1 else ticket[-1]
  
  def create_model(self, test_size):
    # What we want to predict
    y = self.train_data["Survived"]
    # Factors to account on the predictions
    x = self.train_data.drop("Survived", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=101)
    lin_reg = LogisticRegression(max_iter=500000)
    self.lin_reg_model = lin_reg.fit(x_train, y_train)
    return x_test, y_test

  def evaluate(self, dataset):
    return self.lin_reg_model.predict(dataset)


if __name__ == "__main__":
  model = LinearReg("train.csv", "test.csv")

  # Replace NA age and fare by the mean value based on ticket class
  extra_args = {"col_nam": "Age", "model": "train"}
  model.train_data["Age"] = model.train_data[["Age", "Pclass"]].apply(model.calculate_mean_by_class, axis=1, **extra_args)
  extra_args = {"col_nam": "Fare", "model": "train"}
  model.train_data["Fare"] = model.train_data[["Fare", "Pclass"]].apply(model.calculate_mean_by_class, axis=1, **extra_args)

  extra_args = {"col_nam": "Age", "model": "test"}
  model.test_data["Age"] = model.test_data[["Age", "Pclass"]].apply(model.calculate_mean_by_class, axis=1, **extra_args)
  extra_args = {"col_nam": "Fare", "model": "test"}
  model.test_data["Fare"] = model.test_data[["Fare", "Pclass"]].apply(model.calculate_mean_by_class, axis=1, **extra_args)

  # Convert categorical columns
  model.train_data = model.convert_categorical_column("Sex", "train")
  model.test_data = model.convert_categorical_column("Sex", "test")
  model.train_data = model.convert_categorical_column("Embarked", "train")
  model.test_data = model.convert_categorical_column("Embarked", "test")

  # Cleanup Ticker information
  model.train_data["Ticket"] = model.train_data["Ticket"].apply(model.format_ticker_value)
  model.test_data["Ticket"] = model.test_data["Ticket"].apply(model.format_ticker_value)

  extra_args = {"col_nam": "Ticket", "model": "train"}
  model.train_data["Ticket"] = model.train_data[["Ticket", "Pclass"]].apply(model.calculate_mean_by_class, axis=1, **extra_args)
  extra_args = {"col_nam": "Ticket", "model": "test"}
  model.test_data["Ticket"] = model.test_data[["Ticket", "Pclass"]].apply(model.calculate_mean_by_class, axis=1, **extra_args)

  # Drop needless columns
  model.train_data.drop(["Cabin", "Sex", "Embarked", "Name"], axis=1, inplace=True)
  model.test_data.drop(["Cabin", "Sex", "Embarked", "Name"], axis=1, inplace=True)
  model.train_data.dropna(inplace=True)

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
  model.test_data["Survived"] = model.evaluate(model.test_data)

  # Create submission file
  pattern = f"submission_ln*.csv"
  submission_data = model.test_data[["PassengerId", "Survived"]]

  files = glob.glob(pattern)
  last_submission = int(files[-1][-5])
  submission_data.to_csv(f"solutions/submission_ln{str(last_submission+1)}.csv", index=False)
