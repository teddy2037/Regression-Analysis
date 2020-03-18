import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import re

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("network_backup_dataset.csv")
days = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
work_flows = {'work_flow_0':0,'work_flow_1':1,'work_flow_2':2,'work_flow_3':3,'work_flow_4':4}
strip_data = data[['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', \
'Work-Flow-ID', 'File Name','Size of Backup (GB)']].values

def plot_graph_a(a, b, t):
    plt.scatter(a,b, marker = 'x')
    plt.xlabel('True Values', size=15)
    plt.ylabel('Predicted Values', size=15)
    plt.title('Correlation of Predictor: ' + t, size=15)
    plt.draw()
    plt.savefig('D1_lr_a.png', bbox_inches='tight')
    plt.show()

def plot_graph_b(a, b, t):
    plt.scatter(a,b, marker = 'x')
    plt.xlabel('Predicted Values', size=15)
    plt.ylabel('Residuals', size=15)
    plt.title('Error of Predictor: ' + t, size=15)
    plt.draw()
    plt.savefig('D1_lr_b.png', bbox_inches='tight')
    plt.show()

def scalar_encoding(strip_data):
    scal_str = []
    for a,b,c,d,e,f in strip_data:
        red_E = [float(s) for s in re.findall('\d+', e)][0]
        scal_str.append([a, days[b], c, work_flows[d], red_E, f])
    return np.array(scal_str)

def lin_regress(datum):
    kf = KFold(n_splits=10, shuffle = True)
    n = 1

    net_RMSE_trn = 0.0
    net_RMSE_test = 0.0

    print "Fold\tTrain RMSE\tPred. RMSE"
    for train_index, test_index in kf.split(datum):
        X_train = datum[train_index]
        X_test = datum[test_index]
        reg = LinearRegression().fit(X_train[:,:-1], X_train[:,-1])

        rmse_trn = (mean_squared_error(X_train[:,-1], reg.predict(X_train[:,:-1])))
        rmse_pred = (mean_squared_error(X_test[:,-1], reg.predict(X_test[:,:-1])))

        net_RMSE_test = rmse_pred + net_RMSE_test
        net_RMSE_trn = rmse_trn + net_RMSE_trn

        print str(n) + "\t" + ("%.4f" % rmse_trn) + "\t\t" + ("%.4f" % rmse_pred)
        n = n + 1
    print 
    print "Avg. RMSE Training: " + str(sqrt(net_RMSE_trn/10.0))
    print "Avg. RMSE Test: " + str(sqrt(net_RMSE_test/10.0))
    # This linear regression has no hyperparameters to tune in cross validation
    print
    reg_fin = LinearRegression().fit(datum[:,:-1], datum[:,-1])
    pred = reg_fin.predict(datum[:,:-1])
    plot_graph_a(datum[:,-1], pred, "Linear Regression")
    plot_graph_b(pred, (datum[:,-1] - pred), "Linear Regression")

if __name__ == '__main__':
    datum = scalar_encoding(strip_data)
    print
    lin_regress(datum)
    print



