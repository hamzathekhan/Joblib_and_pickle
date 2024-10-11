import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("/content/Social_Network_Ads.csv")
df.info()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index , test_index in split.split(df,df["Purchased"]):
  train_set = df.iloc[train_index]
  test_set = df.iloc[test_index]


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(train_set[["Age","EstimatedSalary"]],train_set["Purchased"])


model.predict([[40,105000]])

# Serielization
import pickle
with open("model_pickle","wb") as f:
  pickle.dump(model,f)
with open("/content/model_pickle", "rb") as f:
  mp = pickle.load(f)
mp.predict([[45,80000]])
