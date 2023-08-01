
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('../static/data/crop_recommendation.csv')

data.isnull().sum()

data['label'].value_counts()

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(data.corr(), annot=True, cmap='viridis')
ax.set(xlabel='features')
ax.set(ylabel='features')

plt.figure(figsize=(15, 8))
plt.subplot(2, 4, 1)
sns.distplot(data['N'], color='blue')
plt.xlabel('Ratio of Nitrogen', fontsize=12)
plt.grid()

plt.subplot(2, 4, 2)
sns.distplot(data['P'], color='green')
plt.xlabel('Ratio of Phosphorous', fontsize=12)
plt.grid()

plt.subplot(2, 4, 3)
sns.distplot(data['K'], color='darkblue')
plt.xlabel('Ratio of Potassium', fontsize=12)
plt.grid()

plt.subplot(2, 4, 4)
sns.distplot(data['temperature'], color='black')
plt.xlabel('Temperature', fontsize=12)
plt.grid()

plt.subplot(2, 4, 5)
sns.distplot(data['rainfall'], color='grey')
plt.xlabel('Rainfall', fontsize=12)
plt.grid()

plt.subplot(2, 4, 6)
sns.distplot(data['humidity'], color='lightgreen')
plt.xlabel('Humidity', fontsize=12)
plt.grid()

plt.subplot(2, 4, 7)
sns.distplot(data['ph'], color='darkgreen')
plt.xlabel('ph level', fontsize=12)
plt.grid()

plt.suptitle('Distribution for Agricultural Conditions', fontsize=20)

x = data.drop(['label'], axis=1)

x = x.values

plt.rcParams['figure.figsize'] = (10, 4)

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++',
                max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize=20)
plt.xlabel('No. of Cluster')
plt.ylabel('wcss')

km = KMeans(n_clusters=4, init='k-means++',
            max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis=1)
z = z.rename(columns={0: 'cluster'})

y = data['label']
x = data.drop(['label'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)


model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

filename = 'crop_pred_model'
pickle.dump(model, open('crop-prediction.pkl', 'wb'))

model = pickle.load(open('crop-prediction.pkl', 'rb'))
