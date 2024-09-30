import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset
crop = pd.read_csv(r"C:\Users\bharg\OneDrive\Desktop\Crop_Recommendation\Crop_recommendation.csv")
print(crop.head())  # Display first few rows
print(crop.shape)  # Print the shape of the dataset
print(crop.info())  # Info on columns, dtypes
print(crop.isnull().sum())  # Check for missing values
print(crop.duplicated().sum())  # Check for duplicates
print(crop.describe())  # Statistical summary

# Drop non-numeric 'label' column for correlation calculation
numeric_crop = crop.drop('label', axis=1)
print(numeric_crop.corr())  # Display correlation matrix

# Plot heatmap of correlation matrix
sns.heatmap(numeric_crop.corr(), annot=True, cbar=True)
plt.show()

# Crop label distribution
print(crop['label'].value_counts())
print(crop['label'].unique().size)

# Plot distributions for 'P' and 'N' using histplot instead of distplot
sns.histplot(crop['P'], kde=True)
plt.show()

sns.histplot(crop['N'], kde=True)
plt.show()

# Crop labels dictionary
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Map crop names to numbers
crop['label'] = crop['label'].map(crop_dict)
print(crop.head())
print(crop['label'].unique())
print(crop['label'].value_counts())

# Splitting data into features and target
X = crop.drop('label', axis=1)
y = crop['label']
print(X.head())
print(y.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler()
X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)

# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Dictionary of classifiers
models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'ExtraTreeClassifier': ExtraTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name} model accuracy: {score}")

# Final RandomForestClassifier model
randclf = RandomForestClassifier()
randclf.fit(X_train, y_train)
y_pred = randclf.predict(X_test)
print("RandomForestClassifier accuracy:", accuracy_score(y_test, y_pred))

# Feature names
print(crop.columns)

# Crop recommendation function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_mx_features = sc.transform(mx_features)
    prediction = randclf.predict(sc_mx_features).reshape(1, -1)
    return prediction[0]

# Example recommendation
N = 90
P = 42
K = 43
temperature = 20.879744
humidity = 82.002744
ph = 6.502985
rainfall = 202.935536
predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended crop label:", predict)

# Save the model, MinMaxScaler, and StandardScaler
pickle.dump(randclf, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))
