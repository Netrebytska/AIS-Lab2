import warnings
warnings.filterwarnings('ignore')
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

X = dataset[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
y = dataset['class']

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()

scatter_matrix(dataset)
pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
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

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

model = SVC(gamma='auto')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Точність класифікації на тестовому наборі:", accuracy_score(y_test, predictions))
print("Матриця помилок на тестовому наборі:\n", confusion_matrix(y_test, predictions))
print("Звіт про класифікацію на тестовому наборі:\n", classification_report(y_test, predictions))


train_predictions = model.predict(X_train)
print("Точність класифікації на тренувальному наборі:", accuracy_score(y_train, train_predictions))
print("Матриця помилок на тренувальному наборі:\n", confusion_matrix(y_train, train_predictions))
print("Звіт про класифікацію на тренувальному наборі:\n", classification_report(y_train, train_predictions))

X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new: {}".format(X_new.shape))

prediction = model.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозована мітка: {}".format(prediction[0]))
print("Ми можемо довіряти моделі на основі її високої точності на тестовому наборі і показників precision, recall і f1-score, які показують хороші результати для всіх класів.")
