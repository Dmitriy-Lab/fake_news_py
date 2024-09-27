from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import sklearn.metrics
import pandas as pd
import seaborn as sns
import numpy as np

# применим функцию read_csv() и посмотрим на первые три записи файла train.csv
train = pd.read_csv('/content/fake_news.csv')
train.head(3)

# Среди новостей половина реальная, а половина - фейковая:
sns.countplot(x = 'label', data = train)

# Переведем значения столбца "label" FAKE и REAL в числовой формат:
pd.get_dummies(train['label'])
label = pd.get_dummies(train['label'], drop_first = True)
label.head(3)

# Добавим новый столбец к нашей базе данных и применим функцию drop() к соответствующим столбцам:

train = pd.concat([train, label], axis = 1)
train.drop(['label'], axis = 1, inplace = True)
train.head(3)

labels = train.REAL
data = train.text  # Список текстовых данных

#Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

# Преобразование текстовых данных в матрицу TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Обучение модели
model = PassiveAggressiveClassifier(max_iter=700)
model.fit(X_train_tfidf, y_train)

# Предсказание категорий для тестовых данных
y_pred = model.predict(X_test_tfidf)

# Предсказание категорий для обучающих данных
y_pred_learn = model.predict(X_train_tfidf)

r = sklearn.metrics.confusion_matrix(y_train, y_pred_learn)
print(f"Матрица ошибок:\n {r}\n")

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy}")
