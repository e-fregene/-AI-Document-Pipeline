import pandas as pd

data_frame = pd.read_csv("src/data/california_housing_test.csv")

"""
Cleaning Data Tutorial
"""
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

#sample DataFrame
data = {
    'Movie': ['Inception', 'Avatar', 'Titanic', 'Joker', 'Batman'],
    'Rating': [8.8, None, 7.8, 8.5, 8.8],
    'Year': [2010, 2009, None, 2019, 2020]
}

data_frame_custom= pd.DataFrame(data)

#Detect missing values
print( "Null values of data", data_frame_custom.isnull())

# Fill missing values with an average or specific value
df_filled = data_frame_custom.fillna({'Rating': data_frame_custom['Rating'].mean(), 'Year': 'Unknown'})

# Drop rows with any missing values
df_dropped = data_frame_custom.dropna()

print("DataFrame after filling missing values:\n", df_filled)
print("DataFrame after dropping rows with missing values:\n", df_dropped)



# Find then Remove duplicates

duplicates = data_frame_custom.duplicated()

print("Duplicate rows:\n", duplicates)

df_no_duplicates = data_frame_custom.drop_duplicates()

print("DataFrame after removing duplicates:\n", df_no_duplicates)


# Correcting data types
data_frame_custom['Year'] = data_frame_custom['Year'].astype(str)

print("DataFrame after converting 'Year' to string:\n", data_frame_custom)


"""
Cleaning on bank_loan dataset
"""

# Load
df = pd.read_csv("src/data/bank_loan.csv")

# Inspect
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nData types:\n", df.dtypes)

#Fix CCAvg: "1/60" -> 1.60
df["CCAvg"] = df["CCAvg"].str.replace("/", ".").astype(float)

# Drop negative Experience values (data entry error)
df = df[df["Experience"] >= 0]

#Drop duplicates (none here)
df = df.drop_duplicates()

# Confirm fix
print("\nCleaned CCAvg sample:", df["CCAvg"].head(5).tolist())
print("Cleaned dtypes:\n", df.dtypes)

# Save cleaned version
df.to_csv("src/data/bank_loan_cleaned.csv", index=False)
print("\nSaved to src/data/bank_loan_cleaned.csv")