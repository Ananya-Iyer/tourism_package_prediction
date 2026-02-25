# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import OneHotEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/ananyaarvindiyer/tourism-package-prediction-ui/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded from hugging face successfully.")

duplicates = df[df["CustomerID"].duplicated()]["CustomerID"].unique()
print(duplicates)

# Drop unique identifier column CustomerID as its not useful for modeling
if not duplicates:
  df.drop(columns=['CustomerID'], inplace=True)

# handle gender inputs having Fe Male as inputs and replace them with female
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
# handle and merge single and unmarried as 1 since both mean the same thing
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})


# Define target variable
target_col = 'ProdTaken'


# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# uploading the datasets back to hugging face
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="ananyaarvindiyer/tourism-package-prediction-ui",
        repo_type="dataset",
    )
