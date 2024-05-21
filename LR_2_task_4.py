import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
           'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
dataset = pd.read_csv(url, names=columns, skipinitialspace=True, na_values="?")

dataset.dropna(inplace=True)

for column in dataset.select_dtypes(include=['object']).columns:
    dataset[column] = dataset[column].astype('category').cat.codes

X = dataset.drop('income', axis=1)
y = dataset['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

models = [
    ('LR', LogisticRegression(solver='liblinear')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

best_model_name, best_model = max(zip(names, [LogisticRegression(solver='liblinear'), LinearDiscriminantAnalysis(),
                                              KNeighborsClassifier(), DecisionTreeClassifier(), GaussianNB(), SVC(gamma='auto')]),
                                  key=lambda x: cross_val_score(x[1], X_train, y_train, cv=kfold, scoring='accuracy').mean())

print(f"Найкраща модель: {best_model_name}")

best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

print("Точність класифікації на тестовому наборі:", accuracy_score(y_test, predictions))
print("Матриця помилок на тестовому наборі:\n", confusion_matrix(y_test, predictions))
print("Звіт про класифікацію на тестовому наборі:\n", classification_report(y_test, predictions))

X_new = np.array([[39, 6, 77516, 13, 4, 1, 0, 1, 4, 1, 2174, 0, 40, 39]])
print("Форма масиву X_new: {}".format(X_new.shape))

prediction = best_model.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозована мітка: {}".format(prediction[0]))
