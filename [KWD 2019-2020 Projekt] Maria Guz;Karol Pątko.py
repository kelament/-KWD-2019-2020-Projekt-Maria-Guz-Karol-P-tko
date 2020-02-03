from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import cross_val_score

# ==================================================================


mse = lambda alg: mean_squared_error(y_test, alg.predict(x_test_poly))
prnt = lambda alg, mse, al: print("MSE( ", alg, " | alpha = ", al, " ):  ", mse, sep="")
doArray = lambda title, algh, alg, X: [alg(title, x, algh) for x in X]
coef = lambda alg: print(alg.coef_)
fit = lambda alg: alg.fit(x_train_poly, y_train)


# ----------------------------------------------------

def MSE(title, alg, al):
    par = mse(alg)
    prnt(title, par, al)
    return par


def TestAlpha(title, al, alg):
    par = alg(alpha=al)
    par.fit(x_train_poly, y_train)
    mse = MSE(title, par, al)
    return mse


def doPlot(*X, title, labelX, labelY):
    fig = plt.figure(figsize=(19,8))
    for (x,y) in X:
        plt.plot(x, label=y)
    plt.legend()
    plt.title(title)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.show()


def doFit(*X):
    for x in X:
        x.fit(x_train_poly, y_train)

def doFitLinear(*X):
    for x in X:
        x.fit(x_train, y_train)


def doGridSearch(parameters, alg):
    model = GridSearchCV(alg, parameters, cv=3)
    model.fit(x_train_poly, y_train)
    print(model.best_params_)
    print(model.best_estimator_)

# ==========================================================


# boston =load_boston()
# print(boston['DESCR'])
# data_desc = pd.DataFrame(boston.data, columns=[boston.feature_names])
# data_desc.describe()
# sb.pairplot(data_desc, diag_kind="kde", palette="husl")
# -------------------------------------------------------STANDARD SCALLER----------------------------------------------


x, y = load_boston(return_X_y=True)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1010)
pf = PolynomialFeatures(2)

x_train_poly = pf.fit_transform(x_train)
x_test_poly = pf.fit_transform(x_test)

# -------------------------------------------------------POLYNOMIAL DEGREE---------------------------------------------

# pf = PolynomialFeatures(2)
# pf3 = PolynomialFeatures(3)
# pf4 = PolynomialFeatures(4)
#
# x_train_poly = pf.fit_transform(x_train)
# x_test_poly = pf.fit_transform(x_test)
#
# x_train_poly3 = pf3.fit_transform(x_train)
# x_test_poly3 = pf3.fit_transform(x_test)
#
# x_train_poly4 = pf4.fit_transform(x_train)
# x_test_poly4 = pf4.fit_transform(x_test)
#
# polynomialF = LinearRegression()
# polynomialF.fit(x_train_poly, y_train)
# mse2train = mean_squared_error(y_train, polynomialF.predict(x_train_poly))
# mse2test = mean_squared_error(y_test, polynomialF.predict(x_test_poly))
# prnt("Second degree polynomial(train)", mse2train)
# prnt("Second degree polynomial(test)", mse2test)
#
# polynomialF = LinearRegression()
# polynomialF.fit(x_train_poly3, y_train)
# mse3train = mean_squared_error(y_train, polynomialF.predict(x_train_poly3))
# mse3test = mean_squared_error(y_test, polynomialF.predict(x_test_poly3))
# prnt("Third degree polynomial(train)", mse3train)
# prnt("Third degree polynomial(test)", mse3test)
#
# polynomialF = LinearRegression()
# polynomialF.fit(x_train_poly4, y_train)
# mse4train = mean_squared_error(y_train, polynomialF.predict(x_train_poly4))
# mse4test = mean_squared_error(y_test, polynomialF.predict(x_test_poly4))
# prnt("Fourth degree polynomial(train)", mse4train)
# prnt("Fourth degree polynomial", mse4test)
#
# doPlot(mse2train,mse3train,mse4train, title="Mean squared error (train)", labelX='Degree of a polynomial', labelY='Train MSE')
#
# doPlot(mse2test,mse3test,mse4test, title="Mean squared error (test)", labelX='Degree of a polynomial', labelY='Test MSE')



# -------------------------------------------------------------REGRESSION--------------------------------------------------

linear = LinearRegression()
polynomialF = LinearRegression()
ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()

doFitLinear(linear)
doFit(polynomialF, ridge, lasso, elastic)

predictLinear = linear.predict(x_test)
predictPolynomialF = polynomialF.predict(x_test_poly)
predictRidge = ridge.predict(x_test_poly)
predictLasso = lasso.predict(x_test_poly)
predictElastic = elastic.predict(x_test_poly)

# --------------------------------------------------------------
doPlot((y_test, 'Test data'), title="Test Data", labelX='Sample', labelY='target')

doPlot((predictLinear, 'Linear'), title="Linear Regression", labelX='Sample', labelY='target')
doPlot((predictPolynomialF, 'Polynomial'), title="Polynomial Features Regression", labelX='Sample', labelY='target')
doPlot((predictRidge, 'Ridge'), title="Ridge Regression", labelX='Sample', labelY='target')
doPlot((predictLasso, 'Lasso'), title="Lasso Regression", labelX='Sample', labelY='target')
doPlot((predictElastic, 'ElasticNet'), title="ElasticNet Regression", labelX='Sample', labelY='target')
doPlot((y_test, 'Test data'), (predictPolynomialF, 'Polynomial'), (predictRidge, 'Ridge'), (predictLasso, 'Lasso'), (predictElastic, 'ElasticNet'), title="Prediction", labelX='Sample',
       labelY='target')


prnt("Linear Regression", mean_squared_error(y_test, predictLinear), "-")
MSE("Polynomial Features Regression", polynomialF, "-")
MSE("Ridge Regression", ridge, "default")
MSE("Lasso Regression", lasso, "default")
MSE("ElasticNet Regression", elastic, "default")
# # --------------------------------------------------------------R2 SCORE-----------------------------------------------------


# score = linear.score(x_test, y_test)
# print("Linear Regression variance score: %.2f" % score)
# score = polynomialF.score(x_test_poly, y_test)
# print("Polynomial Regression variance score: %.2f" % score)
# score = ridge.score(x_test_poly, y_test)
# print("Ridge Regression variance score: %.2f" % score)
# score = lasso.score(x_test_poly, y_test)
# print("Lasso Regression variance score: %.2f" % score)
# score = elastic.score(x_test_poly, y_test)
# print("Elastic Regression variance score: %.2f" % score)
# #-------------------------------------------------------------CROSS VALIDATION-----------------------------------------


# print("CROSS VALIDATION".center(100), "\n")
# scores = cross_val_score(linear, x_train_poly, y_train, cv=10)
# print("Linear Regression: |  ", scores)
# scores = cross_val_score(polynomialF, x_train_poly, y_train, cv=10)
# print("Polynomial Regression: |  ", scores)
# scores = cross_val_score(ridge, x_train_poly, y_train, cv=10)
# print("Ridge Regression: |  ", scores)
# scores = cross_val_score(lasso, x_train_poly, y_train, cv=10)
# print("Lasso Regression: |  ", scores)
# scores = cross_val_score(elastic, x_train_poly, y_train, cv=10)
# print("ElasticNet Regression: |  ", scores)
# #----------------------------------------------------------------------ALPHA-------------------------------------------------


# alpha = [0.1, 0.2, 0.5, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# mseAlphaRidge = doArray("Ridge Regression", Ridge, TestAlpha, alpha)
# mseAlphaLasso = doArray("Lasso Regression", Lasso, TestAlpha, alpha)
# mseAlphaElastic = doArray("ElasticNet Regression", ElasticNet, TestAlpha, alpha)

# doPlot((mseAlphaRidge, 'Ridge'), title="MSE Alpha Ridge", labelX='alpha', labelY='MSE')
# doPlot((mseAlphaLasso, 'Lasso'), title="MSE Alpha Lasso", labelX='alpha', labelY='MSE')
# doPlot((mseAlphaElastic, 'ElasticNet'),  title="MSE Alpha ElasticNet", labelX='alpha', labelY='MSE')
#
# doPlot((mseAlphaRidge, 'Ridge'), (mseAlphaLasso, 'Lasso'), (mseAlphaElastic, 'ElasticNet'), title="MSE", labelX='alpha', labelY='MSE')
#
#


# #----------------------------------------------------------------GRID SEARCH -----------------------------------------------


# doGridSearch([{'alpha': [0.1, 5, 7, 8, 9, 10, 30], 'copy_X': [True, False], 'fit_intercept': [True, False], 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}], ridge)

# doGridSearch([{'alpha': [ 0.1, 0.3, 0.4, 0.9], 'max_iter': [500,1000], 'normalize':[True,False], 'positive':[True, False], 'precompute':[True, False], 'selection':['cyclic', 'random'], 'tol': [1e-4, 1e-6, 1e-2], 'warm_start':[True, False]}], Lasso())

# doGridSearch([{'alpha': [0.1, 0.5, 0.9], 'l1_ratio': [0.1, 0.2, 0.5, 0.9], 'normalize':[True, False], 'positive':[True, False], 'precompute':[True, False], 'selection':['cyclic', 'random'], 'warm_start':[True, False]}], ElasticNet())



# #---------------------------------------------PREDICT v2---------------------------------------------------------------
ridge = Ridge(alpha=9, copy_X=True, fit_intercept=True, max_iter=None, normalize=False,
      random_state=None, solver='svd', tol=0.001)

lasso=Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
      normalize=False, positive=False, precompute=False, random_state=None,
      selection='random', tol=0.01, warm_start=False)

elastic=ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.1,
           max_iter=1000, normalize=False, positive=False, precompute=False,
           random_state=None, selection='random', tol=0.0001, warm_start=True)

doFit( ridge, lasso, elastic)

score = ridge.score(x_test_poly, y_test)
print("Ridge Regression variance score: %.2f" % score)
score = lasso.score(x_test_poly, y_test)
print("Lasso Regression variance score: %.2f" % score)
score = elastic.score(x_test_poly, y_test)
print("Elastic Regression variance score: %.2f" % score)

predictRidge = ridge.predict(x_test_poly)
predictLasso = lasso.predict(x_test_poly)
predictElastic = elastic.predict(x_test_poly)

doPlot((predictRidge, 'Ridge'), title="Ridge Regression", labelX='Sample', labelY='target')
doPlot((predictLasso, 'Lasso'), title="Lasso Regression", labelX='Sample', labelY='target')
doPlot((predictElastic, 'ElasticNet'), title="ElasticNet Regression", labelX='Sample', labelY='target')

print("MSE Ridge: ", mean_squared_error(y_test, predictRidge))
print("MSE Lasso: ", mean_squared_error(y_test, predictLasso))
print("MSE ElasticNet: ", mean_squared_error(y_test, predictElastic))






