import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns  # Библиотека визуализации данных


# dataset_Facebook
data = pd.read_excel("dataset_Facebook.xlsx")  # Подгрузка данных

Leng = data.shape[0]  # Длина данных
for _ in range(1, Leng*5//100):  # Случайные числа заменяются на НаН
    data.iloc[rd.randint(0, Leng-1), rd.randint(0, 4)] = float('Nan')
data = data.dropna()  # Удаление образцов с битыми данными
print(pd.unique(data['Type']))  # Вывод уникальных категориальных признаков
data = pd.get_dummies(data)  # Обработка категориальных признаков
ll = list(data)

from sklearn.preprocessing import MinMaxScaler  # Модуль нормализации данных
mms = MinMaxScaler()
data = mms.fit_transform(data)
data = pd.DataFrame(data)

sns.set(style='whitegrid', context='notebook')
sns.pairplot(data, height=2.5)  # Построение парных графиков разброса между признаками
plt.savefig('PairPlot.png')  # Сохранение графика
plt.close()  # Закрыть график

cm = np.corrcoef(data[:].T)  # Матрица корреляции между признаками
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=ll, xticklabels=ll)  # Построение тепловой карты корреляций
plt.show()

from sklearn.model_selection import train_test_split  # Модуль разделения выборки на тестовую и тренировочную
X, y = data.iloc[:, 6], data.iloc[:, 7]  # Выбор исследуемых признаков
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Определение важности признаков методом случайного леса
from sklearn.ensemble import RandomForestRegressor  # Регрессия методом случайного леса
print('Важность признаков для признака', ll[17])
Xi, yi = data.iloc[:, np.r_[:16+1, 18:len(ll)]], data.iloc[:, 17]
ll_R = ll.copy()  # Копирование списка признаков
del ll_R[17]  # Удаление исследуемого признака
forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)  # n_estimators-количество деревьев в лесу
forest.fit(Xi, yi)  # Подгонка
imp = forest.feature_importances_  # Важность признаков
indices = np.argsort(imp)[::-1]  # Сортировка признаков
for f in range(Xi.shape[1]):  # Вывод важных признаков
    print("%2d) %- *s %f " % (f + 1, 30, ll[indices[f]], imp[indices[f]]))

# Тепловая матрица важности признаков
cm = np.zeros((len(ll), len(ll)))
for f in range(len(ll)):
    Xi, yi = data.iloc[:, np.r_[:f, f+1:len(ll)]], data.iloc[:, f]
    ll_R = ll.copy()
    del ll_R[f]
    forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    forest.fit(Xi, yi)
    imp = forest.feature_importances_
    imp = np.insert(imp, f, -.5)
    cm[:, f] = imp.T
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels=ll, xticklabels=ll)
plt.show()

# Функция построения графика регрессии
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

from sklearn.linear_model import LinearRegression  # Модуль линейной регрессии
slr = LinearRegression()
slr.fit(X_train, y_train)
lin_regplot(X_train, y_train, slr)
plt.xlabel('0')
plt.ylabel('2')
plt.show()

y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, marker='o', s=40)
plt.scatter(y_test_pred, y_test_pred - y_test, marker='s', s=20)
plt.xlabel('Предсказанные значения')
plt.ylabel('Остаток')
plt.hlines(y=0, lw=2, color='red', xmax=1.1, xmin=-0.1)
plt.xlim([-0.1, 1.1])
plt.show()

print("Линейная регрессия")
from sklearn.metrics import mean_squared_error  # Среднеквадратическая ошибка регрессии
print("MSE тренировка: %.3f, тестирование: %.3f" % (mean_squared_error(y_train, y_train_pred)*1000,
                                                    mean_squared_error(y_test, y_test_pred)*1000))
# Ошибка на тренировочных и тестовых данных
from sklearn.metrics import r2_score  # Метрика ошибка регрессии r2
print("R2 тренировка: %.3f, тестирование: %.3f" % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

from sklearn.preprocessing import PolynomialFeatures  # Полиноминальная регрессия
print("Полиноминальная регрессия")
slr2 = LinearRegression()
for i in range(1, 6):
    print("Степень полинома: ", i)
    lin_regplot(X_train, y_train, slr)
    quadratic = PolynomialFeatures(degree=i)
    X_p_train = quadratic.fit_transform(X_train)  # Создаем полином из тренировочных данных
    X_p_test = quadratic.fit_transform(X_test)  # Создаем полином из тестовых данных
    slr2.fit(X_p_train, y_train)  # Подгонка модели под тренировочные данные
    y_p_train_pred = slr2.predict(X_p_train)  # Значения У из тренировочных данных
    y_p_test_pred = slr2.predict(X_p_test)    # Значения У из тестовых данных

    print("MSE тренировка полином: %.3f, тестирование: %.3f" % (mean_squared_error(y_train, y_p_train_pred)*1000,
                                                                mean_squared_error(y_test, y_p_test_pred)*1000))
    print("R2  тренировка полином: %.3f, тестирование: %.3f" % (r2_score(y_train, y_p_train_pred),
                                                                r2_score(y_test, y_p_test_pred)))

    XX = np.arange(-0.01, 1.01, 0.0001)[:, np.newaxis]
    y_quad_fit = slr2.predict(quadratic.fit_transform(XX))  # Значения У для графика
    plt.plot(XX, y_quad_fit)
    plt.show()

i = 15
print("Степень полинома: ", i)
lin_regplot(X_train, y_train, slr)
quadratic = PolynomialFeatures(degree=i)
X_p_train = quadratic.fit_transform(X_train)  # Создаем полином из тренировочных данных
X_p_test = quadratic.fit_transform(X_test)  # Создаем полином из тестовых данных
slr2.fit(X_p_train, y_train)  # Подгонка модели под тренировочные данные
y_p_train_pred = slr2.predict(X_p_train)  # Значения У из тренировочных данных
y_p_test_pred = slr2.predict(X_p_test)    # Значения У из тестовых данных

print("MSE тренировка полином: %.3f, тестирование: %.3f" % (mean_squared_error(y_train, y_p_train_pred)*1000,
                                                            mean_squared_error(y_test, y_p_test_pred)*1000))
print("R2  тренировка полином: %.3f, тестирование: %.3f" % (r2_score(y_train, y_p_train_pred),
                                                            r2_score(y_test, y_p_test_pred)))

XX = np.arange(-0.01, 1.01, 0.0001)[:, np.newaxis]
y_quad_fit = slr2.predict(quadratic.fit_transform(XX))  # Значения У для графика
plt.plot(XX, y_quad_fit)
plt.xlim([-0.1, 1.1])
plt.ylim([-1500, 1500])
plt.show()

print("Случайный лес")
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
# Регрессия случайным лесом
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest .predict(X_test)
print("MSE тренировка: %.3f, тестирование : %.3f" % (mean_squared_error(y_train, y_train_pred)*1000,
                                                     mean_squared_error(y_test, y_test_pred)*1000))
print("R2  тренировка: %.3f, тестирование : %.3f" % (r2_score(y_train, y_train_pred),
                                                     r2_score(y_test, y_test_pred)))

y_forest = forest.predict(XX)  # Значения У для графика
plt.scatter(X, y, c='blue')
plt.plot(XX, y_forest, c='red')
plt.show()
