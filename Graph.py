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
sns.scatterplot(data=df_clean, x='Tavg', y='RR')
plt.title('Corelation Between Temparature Average and RainFall')
plt.show()