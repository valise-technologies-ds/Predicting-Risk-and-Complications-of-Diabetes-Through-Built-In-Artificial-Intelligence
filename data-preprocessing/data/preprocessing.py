import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# loading the datasets
pima=   pd.read_csv("raw/diabetes.csv")
health=   pd.read_csv("raw/diabetes_health.csv")

# getting missing values
print("pima missing values \n",pima.isnull().sum())
missing_cols=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in missing_cols:
    zero_count=(pima[col]==0).sum()
    print(f"{col} : {zero_count} zeros")

# replacing 0s with NaN
for col in missing_cols:
    pima[col]=pima[col].replace(0,np.nan)
print("pima missing values : ",pima.isnull().sum())

# filling na values with median()
for col in missing_cols:
    pima[col]=pima[col].fillna(pima[col].median())
    print(pima[col].median())
print("pima missing values after filling na \n",pima.isnull().sum())

# feature selection
pima_features=['Glucose','BloodPressure','BMI','Age']
X_pima=pima[pima_features]
y_pima=pima['Outcome']

pima_scaler=StandardScaler()
X_pima_scaled=pima_scaler.fit_transform(X_pima)

#check before and after scaling
print(pd.DataFrame(X_pima,columns=pima_features))
print(pd.DataFrame(X_pima_scaled,columns=pima_features))

## splitting the data

X_pima_train,X_pima_test,y_pima_train,y_pima_test=train_test_split(X_pima_scaled,y_pima,test_size=0.2,random_state=42,stratify=y_pima)

#combining features into columns
train_df=pd.DataFrame(X_pima_train,columns=pima_features)
train_df["Outcome"]=y_pima_train.values

test_df=pd.DataFrame(X_pima_test,columns=pima_features)
test_df["Outcome"]=y_pima_test.values

train_df.to_csv("pima_train.csv",index=False)
test_df.to_csv("pima_test.csv",index=False)
print("test and train csv files saved successfully")