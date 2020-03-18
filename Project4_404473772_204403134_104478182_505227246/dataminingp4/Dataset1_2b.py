import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
from Dataset1_2a import scalar_encoding, plot_graph_a, plot_graph_b
from sklearn.tree import export_graphviz


def cross_validate(X, y, n_trees=20, max_features=5,max_depth=4):
    kf = KFold(n_splits=10, shuffle = True)
    oob_total = 0
    RMSE_total_train = 0
    RMSE_total_val = 0
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regr = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, max_features=max_features,
                                     bootstrap=True, oob_score=True,  verbose=0).fit(X_train, y_train)
        oob_total += 1 - regr.oob_score_
        RMSE_total_train += mean_squared_error(y_train, regr.predict(X_train))
        RMSE_total_val += mean_squared_error(y_test, regr.predict(X_test))
    RMSE_total_train = np.sqrt(RMSE_total_train / 10.0)
    RMSE_total_val = np.sqrt(RMSE_total_val / 10.0)
    oob_total = oob_total / 10.0
    return RMSE_total_train, RMSE_total_val, oob_total

def sweep_values(X,y):
    for i in range(1,6):
        rmse_val_scores = []
        oob_scores = []
        for j in range(1,201):
            _, rmse_val, oob = cross_validate(X,y,n_trees=j, max_features=i)
            rmse_val_scores.append(rmse_val)
            oob_scores.append(oob)
        plt.figure(1)
        plt.plot(rmse_val_scores, label="f="+str(i))
        plt.figure(2)
        plt.plot(oob_scores, label="f="+str(i))
    plt.figure(1)
    plt.legend()
    plt.ylabel("Average Test RMSE")
    plt.xlabel("Number of trees")
    plt.savefig("RFR_param_search_rmse")
    plt.figure(2)
    plt.legend()
    plt.ylabel("Out of Bag Error")
    plt.xlabel("Number of trees")
    plt.savefig("RFR_param_search_oob")


def sweep_depth(X,y):
    rmse_val_scores = []
    rmse_train_scores = []
    oob_scores = []
    depths = range(1,21)
    for i in depths:
        rmse_train, rmse_val, oob = cross_validate(X,y, max_features=5, max_depth=i)
        rmse_val_scores.append(rmse_val)
        rmse_train_scores.append(rmse_train)
        oob_scores.append(oob)
        
    plt.plot(depths, rmse_val_scores, label='Test')
    plt.plot(depths, rmse_train_scores, label='Train')
    plt.legend()
    plt.ylabel("Average RMSE")
    plt.xlabel("Max Depth")
    plt.savefig("RFR_depth_rmse")
    plt.figure()
    plt.plot(depths, oob_scores)
    plt.ylabel("Out of Bag Error")
    plt.xlabel("Max Depth")
    plt.savefig("RFR_depth_oob")


if __name__ == '__main__':
    data = pd.read_csv("network_backup_dataset.csv")
    strip_data = data[['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', \
                       'Work-Flow-ID', 'File Name','Size of Backup (GB)']].values
        
    datum = scalar_encoding(strip_data)
    X = datum[:,:-1]
    y = datum[:,-1]
    
    # Part i
    rmse_train, rmse_val, oob = cross_validate(X,y)
    print("Average train RMSE: %f  " % rmse_train)
    print("Average test RMSE: %f " % rmse_val)
    print("Oob Error: %f " % oob)
    
    # Part ii
    sweep_values(X,y)

    # Part iii
    sweep_depth(X,y)
    
    # Part iv
    best_n = 20
    best_f = 5
    best_depth = 8
    rmse_train, rmse_val, oob = cross_validate(X, y, n_trees=20, max_features=5,max_depth=8)
    print("Average train RMSE: %f  " % rmse_train)
    print("Average test RMSE: %f " % rmse_val)
    print("Oob Error: %f " % oob)
    regr = RandomForestRegressor(n_estimators=best_n, max_depth=best_depth, max_features=best_f,
                                 bootstrap=True, oob_score=True).fit(X, y)
                                 
    # plot predictions and residuals
    pred = regr.predict(datum[:,:-1])
    plot_graph_a(datum[:,-1], pred, "Random Forest Regression")
    plot_graph_b(pred, (datum[:,-1] - pred), "Random Forest Regression")
    
    # importances
    print(regr.feature_importances_)

    # Part v - visualize
    feature_names = ['Week #', 'Day of Week', 'Backup Start Time',
        'Work-Flow-ID', 'File Name']
    
    estimator = regr.estimators_[1]
    export_graphviz(estimator, out_file='tree.dot',
                    feature_names = feature_names)







