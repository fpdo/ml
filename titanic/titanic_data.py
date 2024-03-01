
import pandas as pd

class Titanic:
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

  def cleanup(self):
    # Replace NA age and fare by the mean value based on ticket class
    extra_args = {"col_nam": "Age", "model": "train"}
    self.train_data["Age"] = self.train_data[["Age", "Pclass"]].apply(self.calculate_mean_by_class, axis=1, **extra_args)
    extra_args = {"col_nam": "Fare", "model": "train"}
    self.train_data["Fare"] = self.train_data[["Fare", "Pclass"]].apply(self.calculate_mean_by_class, axis=1, **extra_args)

    extra_args = {"col_nam": "Age", "model": "test"}
    self.test_data["Age"] = self.test_data[["Age", "Pclass"]].apply(self.calculate_mean_by_class, axis=1, **extra_args)
    extra_args = {"col_nam": "Fare", "model": "test"}
    self.test_data["Fare"] = self.test_data[["Fare", "Pclass"]].apply(self.calculate_mean_by_class, axis=1, **extra_args)

    # Convert categorical columns
    self.train_data = self.convert_categorical_column("Sex", "train")
    self.test_data = self.convert_categorical_column("Sex", "test")
    self.train_data = self.convert_categorical_column("Embarked", "train")
    self.test_data = self.convert_categorical_column("Embarked", "test")

    # Cleanup Ticker information
    self.train_data["Ticket"] = self.train_data["Ticket"].apply(self.format_ticker_value)
    self.test_data["Ticket"] = self.test_data["Ticket"].apply(self.format_ticker_value)

    extra_args = {"col_nam": "Ticket", "model": "train"}
    self.train_data["Ticket"] = self.train_data[["Ticket", "Pclass"]].apply(self.calculate_mean_by_class, axis=1, **extra_args)
    extra_args = {"col_nam": "Ticket", "model": "test"}
    self.test_data["Ticket"] = self.test_data[["Ticket", "Pclass"]].apply(self.calculate_mean_by_class, axis=1, **extra_args)

    # Drop needless columns
    self.test_passenger_id = self.test_data["PassengerId"]
    self.train_data.drop(["PassengerId", "Cabin", "Sex", "Embarked", "Name"], axis=1, inplace=True)
    self.test_data.drop(["PassengerId", "Cabin", "Sex", "Embarked", "Name"], axis=1, inplace=True)
    self.train_data.dropna(inplace=True)
  
  def append_data(self, model, col_name, col_data):
    dataset = self.train_data
    if model != "train":
      dataset = self.test_data
    
    dataset[col_name] = col_data
