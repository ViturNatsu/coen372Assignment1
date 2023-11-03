from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd



# Method 1: Convert the categorical variables into one-hot encoded vectors
def method1_hotVector(datafram,columns):
    return pd.get_dummies(datafram, columns)


# Method 2: Convert to categories manually
def method2_manual(dataframe, column):
    dataframe[column] = dataframe[column].astype('category')
    return dataframe[column].cat.codes

def train_and_test_base_decision_tree(X_train, X_test, y_train, y_test, dataset_name):
    # Create and train the base Decision Tree model
    base_dt_model = DecisionTreeClassifier()
    base_dt_model.fit(X_train, y_train)

    # Test the model and collect performance metrics
    predictions = base_dt_model.predict(X_test)

    # Calculate performance metrics
    confusion = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)
    f1 = f1_score(y_test, predictions, average=None)
    accuracy = accuracy_score(y_test, predictions)

    # Visualize the decision tree (optional)
    # You can use graphviz or other libraries for tree visualization


def train_and_test_top_decision_tree(X_train, X_test, y_train, y_test, dataset_name):
    # Create and train the top Decision Tree model using grid search
    # Experiment with hyperparameters (criterion, max depth, min samples split)
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],  # Adjust these values as needed
        'min_samples_split': [2, 5, 10]  # Adjust these values as needed
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    top_dt_model = grid_search.best_estimator_

    # Test the model and collect performance metrics
    predictions = top_dt_model.predict(X_test)

    # Calculate performance metrics
    confusion = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)
    f1 = f1_score(y_test, predictions, average=None)
    accuracy = accuracy_score(y_test, predictions)

    # Visualize the decision tree (optional)
    # You can use graphviz or other libraries for tree visualization


def train_and_test_base_mlp(X_train, X_test, y_train, y_test, dataset_name):
    # Create and train the base Multi-Layered Perceptron (MLP) model
    base_mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')
    base_mlp_model.fit(X_train, y_train)

    # Test the model and collect performance metrics
    predictions = base_mlp_model.predict(X_test)

    # Calculate performance metrics
    confusion = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)
    f1 = f1_score(y_test, predictions, average=None)
    accuracy = accuracy_score(y_test, predictions)


def train_and_test_top_mlp(X_train, X_test, y_train, y_test, dataset_name):
    # Create and train the top MLP model using grid search
    # Experiment with hyperparameters (activation function, network architecture, solver)

    param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],  # Adjust these architectures
        'solver': ['adam', 'sgd']
    }

    grid_search = GridSearchCV(MLPClassifier(), param_grid, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    top_mlp_model = grid_search.best_estimator_

    # Test the model and collect performance metrics
    predictions = top_mlp_model.predict(X_test)

    # Calculate performance metrics
    confusion = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)
    f1 = f1_score(y_test, predictions, average=None)
    accuracy = accuracy_score(y_test, predictions)


if __name__ == "__main__":
    # Load the penguins dataset
    df_penguins = pd.read_csv('penguins.csv')

    # Display the first few rows of the dataset
    print(df_penguins.head())

    columns =['species', 'sex', 'island']

    # Method 1: Convert these features into 1-hot vectors
    df_penguins_encoder = method1_hotVector(df_penguins,columns)

    # Method 2: Convert to categories manually
    for column in columns:
        df_penguins[column] = method2_manual(df_penguins, column)

    # Load the penguins dataset
    df_abalone = pd.read_csv('abalone.csv')

    # Display the first few rows of the dataset
    print(df_abalone.head())


    # Method 1: Convert these features into 1-hot vectors
    df_abalone_encoder = method1_hotVector(df_abalone,'Type')

    # Method 2: Convert to categories manually
    df_abalone['Type'] = method2_manual(df_abalone, 'Type')


    # Display the first few rows to verify the changes
    print("Method 1:")
    print(df_penguins_encoder.head())
    print("\nMethod 2:")
    print(df_penguins.head())

    # Display the first few rows to verify the changes
    print("Method 1:")
    print(df_abalone_encoder.head())
    print("\nMethod 2:")
    print(df_abalone.head())

