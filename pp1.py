import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('"C:\Users\Nitha Babu\Downloads\archive\US_Accidents_Dec19.csv"')


print(df.info())
print(df.head())

print(df.isnull().sum())


df.drop(columns=['End_Time', 'Description', 'Number'], inplace=True)


df['Weather_Condition'].fillna(df['Weather_Condition'].mode()[0], inplace=True)


df.dropna(subset=['Start_Time', 'City', 'State'], inplace=True)


df['Start_Time'] = pd.to_datetime(df['Start_Time'])

plt.figure(figsize=(8,5))
sns.countplot(x='Severity', data=df)
plt.title('Accident Severity Distribution')
plt.show()

df['Hour'] = df['Start_Time'].dt.hour


plt.figure(figsize=(10,6))
sns.histplot(df['Hour'], bins=24, kde=False)
plt.title('Accidents by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.show()

top_cities = df['City'].value_counts().nlargest(10)


plt.figure(figsize=(10,6))
sns.barplot(x=top_cities.values, y=top_cities.index)
plt.title('Top 10 Cities with Most Accidents')
plt.xlabel('Number of Accidents')
plt.show()

corr_matrix = df[['Severity', 'Hour']].corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

features = df[['Hour', 'Weather_Condition']]
target = df['Severity']


features = pd.get_dummies(features, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')



