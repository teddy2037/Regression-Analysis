import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler, PolynomialFeatures
import numpy as np
from Dataset1_2a import scalar_encoding
from Dataset1_2d import poly_regress
from Dataset1_2b import cross_validate, sweep_values, sweep_depth

def cross_validate_GBR(X, y, n_trees=600, max_depth=4, lr=0.01):
    kf = KFold(n_splits=10, shuffle = True)
    RMSE_total_train = 0
    RMSE_total_val = 0
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regr = GradientBoostingRegressor(learning_rate=lr, n_estimators=n_trees, max_depth=max_depth ).fit(X_train, y_train)
        
        RMSE_total_train += mean_squared_error(y_train, regr.predict(X_train))
        RMSE_total_val += mean_squared_error(y_test, regr.predict(X_test))
    RMSE_total_train = np.sqrt(RMSE_total_train / 10.0)
    RMSE_total_val = np.sqrt(RMSE_total_val / 10.0)
    return RMSE_total_train, RMSE_total_val

def sweep_GBR(X,y):
    # sweep n_trees
    rmse_val_scores = []
    rmse_train_scores = []
    n_trees = range(100,1001, 50)
    for i in n_trees:
        rmse_train, rmse_val = cross_validate_GBR(X,y, n_trees=i)
        rmse_val_scores.append(rmse_val)
        rmse_train_scores.append(rmse_train)
    
    plt.plot(n_trees, rmse_val_scores, label='Test')
    plt.plot(n_trees, rmse_train_scores, label='Train')
    plt.legend()
    plt.ylabel("Average RMSE")
    plt.xlabel("N Trees")
    plt.savefig("GBR_rmse")
    plt.figure()

    # sweep max depth
    rmse_val_scores = []
    rmse_train_scores = []
    depths = range(1,21)
    for i in depths:
        rmse_train, rmse_val = cross_validate_GBR(X,y, max_depth=i)
        rmse_val_scores.append(rmse_val)
        rmse_train_scores.append(rmse_train)
    
    plt.plot(depths, rmse_val_scores, label='Test')
    plt.plot(depths, rmse_train_scores, label='Train')
    plt.legend()
    plt.ylabel("Average RMSE")
    plt.xlabel("Max Depth")
    plt.savefig("GBR_depth_rmse")

def sweep_lr(X,y):
    # sweep lr
    rmse_val_scores = []
    rmse_train_scores = []
    lr = 2.0 ** np.arange(-10,0)
    for l in lr:
        print(l)
        rmse_train, rmse_val = cross_validate_GBR(X,y, lr=l)
        rmse_val_scores.append(rmse_val)
        rmse_train_scores.append(rmse_train)
    plt.figure()
    plt.semilogx(lr, rmse_val_scores, label='Test')
    plt.semilogx(lr, rmse_train_scores, label='Train')
    plt.legend()
    plt.ylabel("Average RMSE")
    plt.xlabel("lr")
    plt.savefig("GBR_lr")


def sweep_depth_RFR(X,y):
    rmse_val_scores = []
    rmse_train_scores = []
    depths = range(1,21)
    for i in depths:
        rmse_train, rmse_val, _ = cross_validate(X,y, max_features=6, max_depth=i)
        rmse_val_scores.append(rmse_val)
        rmse_train_scores.append(rmse_train)
    plt.figure()
    plt.plot(depths, rmse_val_scores, label='Test')
    plt.plot(depths, rmse_train_scores, label='Train')
    plt.legend()
    plt.ylabel("Average RMSE")
    plt.xlabel("Max Depth")
    plt.savefig("RFR_depth_rmse")



def get_one_hot_features(data):
    ft0 = ['ft1','ft2','ft3','ft4', 'ft5', 'ft6']
    numerical_features  = ['ft1', 'ft2', 'ft3']
    encoded_features    = ['ft4', 'ft5', 'ft6']
    target = data['charges']
    One_HOT = np.array([])
    
    encoder = make_column_transformer(
                                      (StandardScaler(),numerical_features),
                                      (OneHotEncoder(sparse=False), encoded_features),
                                      remainder='passthrough'
                                      )
    X = encoder .fit_transform(data[ft0])
    return X, target

def get_scalar_features(data):
    data = pd.read_csv("insurance_data.csv")
    ft6 = {'southwest' :1, 'southeast' :2, 'northwest' :3, 'northeast' :4}
    ft5 = {'yes' :1, 'no' :2}
    ft4 = {'female' :1, 'male' :2}
    
    strip_data = data[['ft1','ft2','ft3','ft4','ft5','ft6','charges']].values
    scal_str = []
    for a,b,c,d,e,f,g in strip_data:
        scal_str.append([a, b, c, ft4[d], ft5[e],ft6[f], g])
    
    scal_str = np.array(scal_str)

    X = scal_str[:,:-1]
    y = scal_str[:,-1]
    return X, y

def poly_features(Data, degree, scalar=True):
    if scalar:
        X,y = get_scalar_features(Data)
    else:
        X,y = get_one_hot_features(Data)
    poly = PolynomialFeatures(degree)
    new_X = poly.fit_transform(X)
    return new_X, y

if __name__ == '__main__':
    data = pd.read_csv("insurance_data.csv")

    #X, y = get_one_hot_features(data)
    X, y = get_scalar_features(data)
    
    # Random Forest Regression
    sweep_values(X,y)
    sweep_depth_RFR(X,y)
    best_n = 20
    best_f = 6
    best_depth = 4
    rmse_train, rmse_val, oob = cross_validate(X,y, n_trees=best_n, max_features=best_f,max_depth=best_depth)
    print("Average train RMSE: %f  " % rmse_train)
    print("Average test RMSE: %f " % rmse_val)
    print("Oob Error: %f " % oob)

    
    # Gradient Boosting
    sweep_GBR(X,y)
    sweep_lr(X,y)
    best_n = 600
    best_depth = 3
    best_lr = 0.01
    rmse_train, rmse_val = cross_validate_GBR(X,y)
    print("Average train RMSE: %f  " % rmse_train)
    print("Average test RMSE: %f " % rmse_val)
    
    # Polynomial
    Train_RMSE = []
    Test_RMSE = []
    new_Data = np.append(X, np.expand_dims(y,axis=1), axis=1)
    degrees = range(1,6)
    for d in degrees:
        avg_train, avg_test = poly_regress(new_Data, d)
        Train_RMSE.append(avg_train)
        Test_RMSE.append(avg_test)
    plt.figure()
    plt.plot(degrees, Train_RMSE, label="Train")
    plt.plot(degrees, Test_RMSE, label="Test")
    plt.legend()
    plt.xlabel("Degree")
    plt.ylabel("Average RMSE")
    plt.savefig("poly_regr_")
    deg = np.argmin(Test_RMSE)
    print("Best polynomial degree: " + str(deg+1))
    print("Train RMSE: " + str(Train_RMSE[deg]))
    print("Test RMSE: " + str(Test_RMSE[deg]))


    # RFR with polynomial features
    for deg in range(1,6):
        rmse_val_scores = []
        X, y = poly_features(data, deg, True)
        best_n = 20
        best_f = X.shape[1]
        for depth in range(1,11):
            _, rmse_val, _ = cross_validate(X,y, n_trees=best_n, max_features=best_f,max_depth=depth)
            rmse_val_scores.append(rmse_val)
        plt.plot(rmse_val_scores, label="Degree="+str(deg))
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Max Degree')
    plt.savefig('poly+rfr')
