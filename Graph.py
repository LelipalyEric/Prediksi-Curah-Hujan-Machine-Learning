import pandas as pd 
import numpy as np

df = pd.read_csv('climate_data.csv')
print("First 5 rows of the datasets: ")
print(df.head())
print("\n")

print("Data Summary:")
print(df.info())
print("\n")

print("Missing Values in each Column:")
print(df.isnull().sum())
df_clean = df.dropna()
#or kalo niat mau ngurus missing values bisa di isi apagitu yang ngebantu modelnya

#------------------------Graph Areas----------------------------#
import matplotlib.pyplot as plt
import seaborn as sns

#------Data Split Anomali------#
# target_cloumn = 'RR'
# y = df_clean[target_cloumn]
# x = df_clean.drop(columns=[target_cloumn])
# print("Column in X: ", x.columns.tolist())
# print("\nShape of X:", x.shape)
# print("Shape of y: ", y.shape)

sns.set_theme(style="darkgrid")
#------Histogram-------#
# sns.histplot(data=df_clean, x='ss')
# plt.show()
# plt.title("Count SS")

#-----Scatter Plot----#
sns.scatterplot(data=df_clean, x='Tavg', y='RH_avg')
plt.title('Corelation Between Average Temparature and Humidity Average')
plt.show()  


#----Data split----#
from sklearn.model_selection import train_test_split
x = df_clean.drop(columns=['RR', "station_id"])
y = df_clean["RR"]

#50% Training set
x_train, x_temp, y_train, y_temp = train_test_split(
    x,y,
    test_size=0.50,
    random_state=67
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp,y_temp,
    test_size=0.50,
    random_state=67
)


print("==================================================")
print("          DATA SPLIT SUMMARY REPORT               ")
print("==================================================")
print(f"Total Dataset Rows:       {x.shape[0]}\n")
print(f"Training Set (Features):   {x_train.shape}  -> 50% (The Textbook)")
print(f"Training Set (Target):     {y_train.shape}\n")
print(f"Validation Set (Features): {x_val.shape}  -> 25% (The Practice Quiz)")
print(f"Validation Set (Target):   {y_val.shape}\n")
print(f"Testing Set (Features):    {x_test.shape}  -> 25% (The Final Exam)")
print(f"Testing Set (Target):      {y_test.shape}")
print("==================================================")


#-----Box Plot----#
sns.boxplot(x=y_train, color="green")
plt.title("Box plot of training target")
plt.xlabel("Rainfall mm")
plt.show()