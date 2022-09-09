"""
This is a simple ML algorithm for classification made with a framework.
"""

# Import modules needed
import numpy as np  #numpy is used to work with arrays and make processing faster
import pandas as pd #extract data from a file
import seaborn as sns #nicer graphics
import matplotlib.pyplot as plt #graphics
from sklearn.model_selection import train_test_split #get a train and test as params
from sklearn.linear_model import LogisticRegression #framework to make logistic regression
from sklearn.metrics import classification_report #generate report
from sklearn.model_selection import learning_curve
# Ignoring warnings generated
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    df = df = pd.read_csv('datasets/Rice_Cammeo_Osmancik.csv')
    df = df.drop(['Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'], axis=1)

    df['Class'] = df.Class.replace('Cammeo', 1)   # 1 for Cammeo class
    df['Class'] = df.Class.replace('Osmancik', 2) # 2 for Osmancik class

    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lg = LogisticRegression(solver='sag', penalty='l2', max_iter=100)
    lg.fit(X_train, y_train)
    # Get coeficients
    print(lg.coef_) # This is the parameter value
    # Get interceptions
    print(lg.intercept_) # This is the bias value

    y_pred = lg.predict(X_test)
    target = ['Cammeo', 'Osmancik']
    print(classification_report(y_test, y_pred, target_names=target))
    print(f'Accuracy of the logistic regression classifier for test is: {lg.score(X_test, y_test)}')
    print('---------------------------------------------------------------------------------------')

    ######################################################################################
    # Improved model
    lg_2 = LogisticRegression(solver='lbfgs', penalty='none', max_iter=5000)
    lg_2 = lg_2.fit(X_train, y_train)
    y_pred2 = lg_2.predict(X_test)
     # Get coeficients
    print(lg_2.coef_) # This is the parameter value
    # Get interceptions
    print(lg_2.intercept_) # This is the bias value
    target = ['Cammeo', 'Osmancik']
    print(classification_report(y_test, y_pred2, target_names=target))
    print(f'Scaled logistic legression accuracy for test is: {lg_2.score(X_test, y_test)}')

    ######################################################################################
    # Validation
    print('---------------------------------------------------------------------------------------')
    predict = lg_2.predict([[15231,525.5789795,229.7498779,85.09378815,0.928882003,15617,0.572895527]])
    print(f"Validate prediction: {predict}")

    ######################################################################################
    # Graph Model 1
    train_sizes, train_scores, test_scores = learning_curve(lg, X, y)
    # Mean and STD of the train scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    # Mean and STD of the test scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, "o-", color="r", label='Trainig score')
    plt.plot(train_sizes, test_mean, "o-", color="g", label='Cross-validation score')
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    ######################################################################################
    # Graph Model 2
    train_sizes, train_scores, test_scores = learning_curve(lg_2, X, y)
    # Mean and STD of the train scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    # Mean and STD of the test scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, "o-", color="r", label='Trainig score')
    plt.plot(train_sizes, test_mean, "o-", color="g", label='Cross-validation score')
    plt.legend(loc="best")
    plt.grid()
    plt.show()



    

