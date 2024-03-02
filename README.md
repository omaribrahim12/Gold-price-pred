Gold Price Prediction

This Python script performs the prediction of gold prices using various regression models and visualizes the data distribution. Below are the details of each section of the code:
Libraries

    pandas: Used for handling data in the form of DataFrames.
    numpy: Utilized for numerical computations.
    matplotlib.pyplot: Used for data visualization.
    seaborn: Another data visualization library based on Matplotlib.
    LinearRegression, LogisticRegression, RandomForestRegressor, SVC, DecisionTreeClassifier: Classification and regression models from Scikit-learn.
    metrics, confusion_matrix, classification_report, train_test_split: Evaluation metrics and data splitting tools from Scikit-learn.

Data Preprocessing

    Reads the dataset gold_data.csv into a DataFrame.
    Displays basic information about the dataset like shape, sample data, information, and statistical summary.

Data Cleaning

    Checks for null values and duplicated values in the dataset.

Data Visualization

    Plots the time series of gold (GLD) and silver (SLV) prices.
    Plots the distribution of the EUR/USD exchange rate.
    Displays distribution plots for gold (GLD), silver (SLV), EUR/USD, and S&P 500 (SPX) prices.

Data Splitting

    Splits the dataset into features (x) and the target variable (y).
    Splits the data into training and testing sets.

Model Training and Evaluation

    Trains a Random Forest Regressor model using the training data.
    Makes predictions on the test data and calculates evaluation metrics such as R-squared error and Mean Squared Error (MSE).
    Visualizes the actual gold prices vs. the predicted prices using a line plot.

Note

    The dataset file (gold_data.csv) is assumed to be present in the same directory as the script.

This script provides insights into gold price prediction using regression techniques and demonstrates the importance of data visualization in understanding the dataset.
