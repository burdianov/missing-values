import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("./data/melb_data.csv")

y = data.Price

melb_predictors = data.drop(["Price"], axis=1)
X = melb_predictors.select_dtypes(exclude=["object"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)
