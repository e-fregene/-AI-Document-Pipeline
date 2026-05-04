import pandas as pd

data_frame = pd.read_csv("california_housing_test.csv")

#Understand data
print(f"This is .head() {data_frame.head()}") # Shows first 5 rows
print(f"This is .info() {data_frame.info()}")
print(f"This is .describe() {data_frame.describe()}")
print(f"This is .shape {data_frame.shape}")


#Sorting Data

print(f"Sorting data by median housing age{data_frame.sort_values(by='housing_median_age')}")

#Missing values
print(data_frame['housing_median_age'].fillna(data_frame['housing_median_age'].mean(), inplace=True))

print(data_frame.dropna(inplace=True)) #remove rows with missing values