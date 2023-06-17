
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Set local path prefix in the processing container
input_data_path = os.path.join(r"/opt/ml/processing/input", "healthcare-dataset-stroke-data.csv")

# df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
df = pd.read_csv(input_data_path)
df.dropna(inplace=True)
df = df[df['gender']!='Other']

df = pd.concat([df, pd.get_dummies(df[['gender', 'work_type', 'Residence_type', 'smoking_status']], drop_first=True)], axis=1)
df['ever_married'] = np.where(df['ever_married']=='Yes', 1, 0)
df['work_type_Never_worked'] = df['work_type_Never_worked'] + df['work_type_children']
               
df.drop(['id', 'gender', 'work_type', 'Residence_type', 'smoking_status', 'work_type_children'], axis=1, inplace=True)

df = pd.concat([df['stroke'], df.drop(['stroke'], axis=1)], axis=1)

print("Shape of data is:", df.shape)
train, test = train_test_split(df, test_size=0.2)
test, validation = train_test_split(test, test_size=0.5)

# try:
#     os.makedirs("/opt/ml/processing/train")
#     os.makedirs("/opt/ml/processing/validation")
#     os.makedirs("/opt/ml/processing/test")
#     # os.makedirs(f"{bucket}/output/train")
#     # os.makedirs(f"{bucket}/output/validation")
#     # os.makedirs(f"{bucket}/output/test")
#     print("Successfully created directories")
# except Exception as e:
#     # if the Processing call already creates these directories (or directory otherwise cannot be created)
#     print(e)
#     print("Could not make directories")
#     pass

# try:
train.to_csv(r"/opt/ml/processing/output/train/train.csv", index=False)
validation.to_csv(r"/opt/ml/processing/output/validation/validation.csv", index=False)
test.to_csv(r"/opt/ml/processing/output/test/test.csv", index=False)
df.to_csv(r"/opt/ml/processing/output/full/df.csv", index=False)

    # train.to_csv(f"{bucket}/output/train/train.csv")
    # validation.to_csv(f"{bucket}/output/validation/validation.csv")
    # test.to_csv(f"{bucket}/output/test/test.csv")
#     print("Wrote files successfully")
# except Exception as e:
#     print("Failed to write the files")
#     print(e)
#     pass

# print("Completed running the processing job")
