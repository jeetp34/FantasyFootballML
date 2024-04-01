import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

os.chdir(r"college_receiving")
folder_path = os.getcwd()
all_files = os.listdir(folder_path)

rookiedf = pd.read_csv("2023.csv", skiprows = 1)
rookiedf.drop(rookiedf.columns[2:23], axis=1, inplace=True)
#draftdf.drop(draftdf.columns[7:17], axis=1, inplace=True)
#rookiedf.insert(9,"Success",0)
drafttest = rookiedf.copy()
drafttest.drop(drafttest.columns[0:2], axis = 1, inplace = True)
print(rookiedf.head())
print(drafttest)


csv_files = [f for f in all_files if f.endswith('.csv') and f != "combined_file.csv" and f != "2023.csv"]

# Create a list to hold the dataframes
df_list = []

for csv in csv_files:
    file_path = os.path.join(folder_path, csv)
    try:
        # Try reading the file using default UTF-8 encoding
        df = pd.read_csv(file_path, skiprows=1)
        df_list.append(df)
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
            df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
            df_list.append(df)
        except Exception as e:
            print(f"Could not read file {csv} because of error: {e}")
    except Exception as e:
        print(f"Could not read file {csv} because of error: {e}")

# Concatenate all data into one DataFrame
combdata = pd.concat(df_list, ignore_index=True)

# Save the final result to a new CSV file
combdata.to_csv(os.path.join(folder_path, 'combined_file.csv'), index=False)

combdata.drop(combdata.columns[3:23], axis=1, inplace=True)
#combdata.drop(combdata.columns[7:17], axis=1, inplace=True)
combdata.insert(10,"Success",0)
#combdata["Name"] = combdata["Name"].str.replace("*", "")
print(combdata.head())

proWR = ["Noah Brown", "Chris Godwin", "Cooper Kupp", "JuJu Smith-Schuster", "Curtis Samuel", "Mike Williams", "Christian Kirk", "Courtland Sutton", "Calvin Ridley", "D.J. Moore", "Terry McLaurin", "DK Metcalf", "A.J. Brown", "Deebo Samuel", "Marquise Brown", "Michael Pittman", "Tee Higgins", "Brandon Aiyuk", "Justin Jefferson", "CeeDee Lamb", "Jerry Jeudy", "Amon-Ra St. Brown", "Nico Collins", "Ja'Marr Chase", "Romeo Doubs", "Christian Watson", "Chris Olave", "Garrett Wilson", "Drake London", "Puka Nacua", "Dontayvion Wicks", "Rashee Rice", "Jordan Addison", "Zay Flowers", "Nathaniel Dell", "Jaylen Waddle", "DeVonta Smith"]


for WR in proWR:
    # Check if the player exists in the DataFrame
    if WR in combdata["Name"].values:
        # If the player exists, get its index
        row_index = combdata[combdata["Name"] == WR].index[0]
        # Update the "Success" column for that player
        combdata.iloc[row_index, 10] = 1
    else:
        # If the player doesn't exist, print a warning message
        print(f"Player '{WR}' not found in DataFrame.")

X = combdata.iloc[:, 3:10]
print(X)
y = combdata.iloc[:, 10]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose a classification model (e.g., logistic regression)
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict_proba(X_test_scaled)
y_test_binary = (y_test > 0).astype(int)
print(classification_report(y_test_binary, np.argmax(y_pred, axis=1)))

print(drafttest)
new_data_scaled = scaler.transform(drafttest)
probability_scores = model.predict_proba(new_data_scaled)

probability_success = probability_scores[:, 1]

# Add the probability scores to the DataFrame
drafttest['Probability_Success'] = probability_success

# Filter the DataFrame to show only candidates predicted to have "success"
predicted_success_candidates = drafttest[drafttest['Probability_Success'] > 0.5]  # You can adjust the threshold as needed

# Display the predicted successful candidates
drafttest = drafttest.join(rookiedf["Name"])

draftsorted = drafttest.sort_values(by='Probability_Success', ascending=False)


os.chdir(r"wr_completed")

draftsorted.to_csv(os.path.join(folder_path, 'wr_completed/draftsorted.csv'), index=False)