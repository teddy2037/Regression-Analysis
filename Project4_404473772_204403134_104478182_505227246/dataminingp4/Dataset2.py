import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import re
from Dataset1_2a import *
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm
from scipy import stats




def plot_graph_a(a, b, t, name = False):
    plt.scatter(a,b, marker = 'x')
    plt.xlabel('True Values', size=15)
    plt.ylabel('Predicted Values', size=15)
    plt.title('Correlation of Predictor: ' + t, size=15)
    if name != False:
        plt.draw()
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def plot_graph_b(a, b, t, name = False):
    plt.scatter(a,b, marker = 'x')
    plt.xlabel('Predicted Values', size=15)
    plt.ylabel('Residuals', size=15)
    plt.title('Error of Predictor: ' + t, size=15)
    if name != False:
        plt.draw()
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def lin_regress_r(datum,coeff = False,pic = False,figname1 = False,figname2 = False):
    kf = KFold(n_splits=10, shuffle = True)
    n = 1

    net_RMSE_trn = 0.0
    net_RMSE_test = 0.0

    # print("Fold\tTrain RMSE\tPred. RMSE")
    for train_index, test_index in kf.split(datum):
        X_train = datum[train_index]
        X_test = datum[test_index]
        reg = LinearRegression().fit(X_train[:,:-1], X_train[:,-1])

        rmse_trn = (mean_squared_error(X_train[:,-1], reg.predict(X_train[:,:-1])))
        rmse_pred = (mean_squared_error(X_test[:,-1], reg.predict(X_test[:,:-1])))

        net_RMSE_test = rmse_pred + net_RMSE_test
        net_RMSE_trn = rmse_trn + net_RMSE_trn

        # print(str(n) + "\t" + ("%.4f" % rmse_trn) + "\t\t" + ("%.4f" % rmse_pred))
        n = n + 1
    print()
    print("Avg. RMSE Training: " + str(sqrt(net_RMSE_trn / 10.0)))
    print("Avg. RMSE Test: " + str(sqrt(net_RMSE_test / 10.0)))
    # This linear regression has no hyperparameters to tune in cross validation

    reg_fin = LinearRegression().fit(datum[:,:-1], datum[:,-1])
    # print()
    if coeff == True:
        for c in reg_fin.coef_:
            print('%.4f'%c)
            print()
    pred = reg_fin.predict(datum[:,:-1])
    if pic == True:
        plot_graph_a(datum[:,-1], pred, "Linear Regression",figname1)
        plot_graph_b(pred, (datum[:,-1] - pred)**2, "Linear Regression",figname2)

def lin_regress_ridge(datum,alpha,coeff = False,pic = False,figname1 = False,figname2 = False):
    kf = KFold(n_splits=10, shuffle = True)
    n = 1

    net_RMSE_trn = 0.0
    net_RMSE_test = 0.0

    # print("Fold\tTrain RMSE\tPred. RMSE")
    for train_index, test_index in kf.split(datum):
        X_train = datum[train_index]
        X_test = datum[test_index]
        reg = Ridge(alpha=alpha).fit(X_train[:,:-1], X_train[:,-1])

        rmse_trn = (mean_squared_error(X_train[:,-1], reg.predict(X_train[:,:-1])))
        rmse_pred = (mean_squared_error(X_test[:,-1], reg.predict(X_test[:,:-1])))

        net_RMSE_test = rmse_pred + net_RMSE_test
        net_RMSE_trn = rmse_trn + net_RMSE_trn

        # print(str(n) + "\t" + ("%.4f" % rmse_trn) + "\t\t" + ("%.4f" % rmse_pred))
        n = n + 1
    # print()
    print("Avg. RMSE Training: " + str(sqrt(net_RMSE_trn / 10.0)))
    print("Avg. RMSE Test: " + str(sqrt(net_RMSE_test / 10.0)))
    # This linear regression has no hyperparameters to tune in cross validation
    # print()
    reg_fin = Ridge(alpha=alpha).fit(datum[:,:-1], datum[:,-1])
    pred = reg_fin.predict(datum[:,:-1])
    if coeff == True:
        for c in reg_fin.coef_:
            print('%.4f'%c)
            print()
    if pic == True:
        plot_graph_a(datum[:,-1], pred, "Ridge")
        plot_graph_b(pred, (datum[:,-1] - pred)**2, "Ridge")
    return sqrt(net_RMSE_test / 10.0)


def lin_regress_lasso(datum,alpha,coeff = False,pic = False,figname1 = False,figname2 = False):
    kf = KFold(n_splits=10, shuffle = True)
    n = 1

    net_RMSE_trn = 0.0
    net_RMSE_test = 0.0

    # print("Fold\tTrain RMSE\tPred. RMSE")
    for train_index, test_index in kf.split(datum):
        X_train = datum[train_index]
        X_test = datum[test_index]
        reg = Lasso(alpha=alpha).fit(X_train[:,:-1], X_train[:,-1])

        rmse_trn = (mean_squared_error(X_train[:,-1], reg.predict(X_train[:,:-1])))
        rmse_pred = (mean_squared_error(X_test[:,-1], reg.predict(X_test[:,:-1])))

        net_RMSE_test = rmse_pred + net_RMSE_test
        net_RMSE_trn = rmse_trn + net_RMSE_trn

        # print(str(n) + "\t" + ("%.4f" % rmse_trn) + "\t\t" + ("%.4f" % rmse_pred))
        n = n + 1
    # print()
    print("Avg. RMSE Training: " + str(sqrt(net_RMSE_trn / 10.0)))
    print("Avg. RMSE Test: " + str(sqrt(net_RMSE_test / 10.0)))
    # # This linear regression has no hyperparameters to tune in cross validation

    reg_fin = Lasso(alpha=alpha).fit(datum[:,:-1], datum[:,-1])
    pred = reg_fin.predict(datum[:,:-1])
    if coeff == True:
        for c in reg_fin.coef_:
            print('%.4f'%c)
            print()
    if pic == True:
        plot_graph_a(datum[:,-1], pred, "Lasso",figname1)
        plot_graph_b(pred, (datum[:,-1] - pred)**2, "Lasso",figname2)
    return sqrt(net_RMSE_test / 10.0)



def lin_regress_elast(datum,alpha,l1_ratio,coeff = False,pic = False,figname1 = False,figname2 = False):
    kf = KFold(n_splits=10, shuffle = True)
    n = 1

    net_RMSE_trn = 0.0
    net_RMSE_test = 0.0

    # print("Fold\tTrain RMSE\tPred. RMSE")
    for train_index, test_index in kf.split(datum):
        X_train = datum[train_index]
        X_test = datum[test_index]
        reg = ElasticNet(alpha=alpha,l1_ratio=l1_ratio).fit(X_train[:,:-1], X_train[:,-1])

        rmse_trn = (mean_squared_error(X_train[:,-1], reg.predict(X_train[:,:-1])))
        rmse_pred = (mean_squared_error(X_test[:,-1], reg.predict(X_test[:,:-1])))

        net_RMSE_test = rmse_pred + net_RMSE_test
        net_RMSE_trn = rmse_trn + net_RMSE_trn

        # print(str(n) + "\t" + ("%.4f" % rmse_trn) + "\t\t" + ("%.4f" % rmse_pred))
        n = n + 1
    # print()
    print("Avg. RMSE Training: " + str(sqrt(net_RMSE_trn / 10.0)))
    print("Avg. RMSE Test: " + str(sqrt(net_RMSE_test / 10.0)))
    # This linear regression has no hyperparameters to tune in cross validation

    reg_fin = Lasso(alpha=alpha).fit(datum[:,:-1], datum[:,-1])
    pred = reg_fin.predict(datum[:,:-1])
    if coeff == True:
        for c in reg_fin.coef_:
            print('%.4f'%c)
            print()
    if pic == True:
        plot_graph_a(datum[:,-1], pred, "Lasso",figname1)
        plot_graph_b(pred, (datum[:,-1] - pred)**2, "Lasso",figname2)
    return sqrt(net_RMSE_test / 10.0)


if __name__ == '__main__':
    catname = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    featuresdata = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    target = ['MEDV']

    enc = OneHotEncoder(handle_unknown='ignore',categories='auto')
    df = pd.read_csv("housing_data.csv", names= catname, header=None)
    X = df.loc[:,featuresdata].values
    y = df.loc[:,target].values
    dk = df.values

    print()
    lin_regress_r(dk, pic=True, figname1='datset2q11.png',figname2='datset2q12.png')

    print()







    lin_regress_r(dk)

    # print("Pavan code")
    # print()
    # lin_regress(dk)
    print()
    kmin_ridge = 1e9
    kmin_lasso = 1e9

    # c = [1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4,1e5]

    best_c_ridge = 0
    best_c_lasso = 0

    #
    # print('without one hot ecnoding')

    # c = [1e-4,1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    c = np.logspace(-5,5,20)
    kstore1 = []
    kstore2 = []

    kmin_ridge = 1e9
    kmin_lasso = 1e9
    best_c_ridge = 0
    best_c_lasso = 0

    print('Ridge')

    for alpha in c:
        # print(alpha)
        # print()
        k = lin_regress_ridge(dk,alpha)
        kstore1.append(k)
        if k <kmin_ridge:
            kmin_ridge = k
            best_c_ridge = alpha
        # print()

    print('Lasso')
    for alpha in c:
        # print(alpha)
        # print()
        k = lin_regress_ridge(dk,alpha)
        kstore2.append(k)
        if k < kmin_lasso:
            kmin_lasso = k
            best_c_lasso = alpha
        # print()
    print()
    print('Ridge alpha ',best_c_ridge,' RMSE ',kmin_ridge)
    print()
    print('Lasso alpha ', best_c_lasso, ' RMSE ', kmin_lasso)


    plt.plot(c,kstore1)
    plt.xscale('log')

    plt.xlabel('Alpha', size=15)
    plt.ylabel('Test RMSE', size=15)
    plt.title('Test RMSE vs Alpha', size=15)
    plt.draw()
    plt.savefig('d2ridge.png', bbox_inches='tight')
    plt.show()

    plt.plot(c,kstore2)
    plt.xscale('log')

    plt.xlabel('Alpha', size=15)
    plt.ylabel('Test RMSE', size=15)
    plt.title('Test RMSE vs Alpha', size=15)
    plt.draw()
    plt.savefig('d2lasso.png', bbox_inches='tight')
    plt.show()







    ###########################   elastic net ###################################################
    a = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    b = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    alpha = []
    l1_ratio = []

    for i in a:
        for j in b:
            alpha.append(i + j)
            l1_ratio.append(i / (i + j))

    kmin_net = 1e9
    best_a_net = 0
    best_l1_net = 0

    kstore3 = []
    for index, a in enumerate(alpha):
        # print(a)
        # print()
        k = lin_regress_ridge(dk, a, l1_ratio[index])
        kstore3.append(k)
        if k < kmin_net:
            kmin_net = k
            best_a_net = a
            best_l1_net = l1_ratio[index]
        # print()
    print()
    print('L1 ',best_a_net * best_l1_net,' L2 ',best_a_net * (1 - best_l1_net), ' RMSE', kmin_net)



    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())


    print('-'*35)
    print('ridge')
    lin_regress_ridge(dk, best_c_ridge,coeff=True)
    print('-' * 35)
    print('lasso')
    lin_regress_lasso(dk,best_c_lasso,coeff=True)
    print('-' * 35)
    print('elastic net')
    lin_regress_elast(dk,best_a_net,best_l1_net,coeff=True)
    print('-' * 35)
    print('linear')
    lin_regress_r(dk,coeff=True)
    print('-' * 35)


############################### testing ##################################
    catname = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    featuresdata = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    target = ['MEDV']
    for index in range(13):
        featuresdatak = featuresdata[: index] + featuresdata[index + 1:]
        print(featuresdatak)
        df = pd.read_csv("housing_data.csv", names=catname, header=None)
        X = df.loc[:, featuresdatak].values
        y = df.loc[:, target].values
        dk = df.values
        print()
        lin_regress_r(dk)











