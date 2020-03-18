import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from Dataset1_2a import scalar_encoding
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def poly_regress(Data, degree):
    poly = PolynomialFeatures(degree)
    X = Data[:,:-1]
    y = Data[:,-1]
    new_X = poly.fit_transform(X)
    new_Data = np.append(new_X, np.expand_dims(y,axis=1), axis=1)
    return lin_regress(new_Data)


def lin_regress(datum):
    # Pavan's function, without the graphs
    kf = KFold(n_splits=10, shuffle = True)
    mse_trn = 0.0
    mse_pred = 0.0
    
    for train_index, test_index in kf.split(datum):
        X_train = datum[train_index]
        X_test = datum[test_index]
        reg = LinearRegression().fit(X_train[:,:-1], X_train[:,-1])
        
        mse_trn += mean_squared_error(X_train[:,-1], reg.predict(X_train[:,:-1]))
        mse_pred += mean_squared_error(X_test[:,-1], reg.predict(X_test[:,:-1]))
 
    return mse_trn/10.0, mse_pred/10.0

if __name__ == '__main__':
    data = pd.read_csv("network_backup_dataset.csv")
    strip_data = data[['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', \
                       'Work-Flow-ID', 'File Name','Size of Backup (GB)']].values
        
    datum = scalar_encoding(strip_data)
    # Split by workflow
    Data = []
    for i in range(5):
        indices = (datum[:,3] == i)
        Data.append(datum[indices, :][:, [0,1,2,4,5]])
    
    # Part i
    for i in range(5):
        print("Workflow " + str(i))
        avg_train, avg_test = lin_regress(Data[i])
        print("Avg. RMSE Training: " + str(np.sqrt(avg_train)))
        print("Avg. RMSE Test: " + str(np.sqrt(avg_test)))


    # Plots
    for i in range(5):
        X = Data[i][:,:-1]
        y = Data[i][:,-1]
        regr = LinearRegression().fit(X, y)
        pred = regr.predict(X)
        plt.figure(1)
        plt.scatter(y, pred, marker = 'x', label='Workflow ' + str(i))
        plt.figure(2)
        plt.scatter(pred, y - pred, marker = 'x', label='Workflow ' + str(i))
        
    plt.figure(1)
    plt.legend()
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.xlabel('True Values', size=15)
    plt.ylabel('Predicted Values', size=15)
    plt.title('Predicted Values: Separate Workflows', size=15)
    plt.savefig('Dataset1_d_fitted.png', bbox_inches='tight')
        
    plt.figure(2)
    plt.legend()
    plt.xlabel('Predicted Values', size=15)
    plt.ylabel('Residuals', size=15)
    plt.title('Error of Predictor:  Separate Workflows', size=15)
    plt.savefig('Dataset1_d_residuals.png', bbox_inches='tight')

    # Part ii - fit polynomial
    print("\nUsing polynomial features:")
    # search for optimal degree
    for i in range(5):
        Train_RMSE = []
        Test_RMSE = []
        degrees = range(1,16)
        for d in degrees:
            avg_train, avg_test = poly_regress(Data[i], d)
            Train_RMSE.append(np.sqrt(avg_train))
            Test_RMSE.append(np.sqrt(avg_test))
        plt.figure()
        plt.plot(degrees, Train_RMSE, label="Train")
        plt.plot(degrees, Test_RMSE, label="Test")
        plt.legend()
        plt.xlabel("Degree")
        plt.ylabel("Average RMSE")
        plt.savefig("poly_regr_" + str(i))

    # get RMSE of optimal model
    best_degrees = [8, 9, 9, 8, 9]
    for i in range(5):
        print("Workflow " + str(i))
        avg_train, avg_test = poly_regress(Data[i], best_degrees[i])
        print("Avg. RMSE Training: " + str(np.sqrt(avg_train)))
        print("Avg. RMSE Test: " + str(np.sqrt(avg_test)))

    # Plots
    for i in range(5):
        poly = PolynomialFeatures(best_degrees[i])
        X = Data[i][:,:-1]
        y = Data[i][:,-1]
        new_X = poly.fit_transform(X)
        regr = LinearRegression().fit(new_X, y)
        pred = regr.predict(new_X)
        plt.figure(3)
        plt.scatter(y, pred, marker = 'x', label='Workflow ' + str(i))
        plt.figure(4)
        plt.scatter(pred, y - pred, marker = 'x', label='Workflow ' + str(i))

    plt.figure(3)
    plt.legend()
    plt.xlabel('True Values', size=15)
    plt.ylabel('Predicted Values', size=15)
    plt.title('Predicted Values: Polynomial Regression', size=15)
    plt.savefig('Dataset1_d_fitted_poly.png', bbox_inches='tight')

    plt.figure(4)
    plt.legend()
    plt.xlabel('Predicted Values', size=15)
    plt.ylabel('Residuals', size=15)
    plt.title('Correlation of Predictor: Polynomial Regression', size=15)
    plt.savefig('Dataset1_d_residuals_poly.png', bbox_inches='tight')
