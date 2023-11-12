import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_plot(conf_matrix, title, filename):
    plt.figure(figsize=(6, 4))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(conf_matrix))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)


penguin_data = pd.read_csv('penguins.csv')
abalone_data = pd.read_csv('abalone.csv')

# Print abalone dataset's first 5 rows
print(abalone_data.head())
print(abalone_data.columns)

# Print Penguin dataset's first 5 rows
print(penguin_data.head())
print(penguin_data.columns)

# -METHOD 1-  One-hot vector
penguin_data = pd.get_dummies(penguin_data, columns=['island', 'sex'])

# -METHOD2-  CATEGORY FOR CONVERTING CATEGORIES
# mapping_islands = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 11}
# mapping_sex = {'MALE': 0, 'FEMALE': 1}
#
# penguin_data['island'] = penguin_data['island'].map(mapping_islands)
# penguin_data['sex'] = penguin_data['sex'].map(mapping_sex)
# penguin_data.head()


penguin_class_distribution = penguin_data['species'].value_counts()
penguin_class_percentages = (penguin_class_distribution / len(penguin_data)) * 100
penguin_class_percentages.plot(kind='bar', title='Percentage of Instances in Penguin Class Distribution', color='blue')
plt.ylabel('Percentage')
plt.xlabel('Species')
plt.savefig('penguin-classes.png')
plt.close()

abalone_class_distribution = abalone_data['Type'].value_counts()
abalone_class_percentages = (abalone_class_distribution / len(abalone_data)) * 100
abalone_class_percentages.plot(kind='bar', title='Percentage of Instances in Abalone Class Distribution', color='red')
plt.ylabel('Percentage')
plt.xlabel('Type')
plt.savefig('abalone-classes.png')
plt.close()

penguin_X = penguin_data.drop('species', axis=1)
penguin_y = penguin_data['species']
abalone_X = abalone_data.drop('Type', axis=1)
abalone_y = abalone_data['Type']

penguin_X_train, penguin_X_test, penguin_y_train, penguin_y_test = train_test_split(penguin_X, penguin_y, test_size=0.2,
                                                                                    random_state=42)
abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test = train_test_split(abalone_X, abalone_y, test_size=0.2,
                                                                                    random_state=42)

penguin_base_dt = tree.DecisionTreeClassifier()
penguin_base_dt.fit(penguin_X_train, penguin_y_train)
tree.plot_tree(penguin_base_dt, filled=True, rounded=True, feature_names=penguin_X.columns, class_names=penguin_y.unique())
plt.savefig('penguin-base-dt.png', dpi=400)
plt.close()

abalone_base_dt = DecisionTreeClassifier(max_depth=3)
abalone_base_dt.fit(abalone_X_train, abalone_y_train)
plot_tree(abalone_base_dt, filled=True, rounded=True, feature_names=abalone_X.columns, class_names=abalone_y.unique())
plt.savefig('abalone-base-dt.png', dpi=400)
plt.close()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search_penguin_dt = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='accuracy', cv=5)
grid_search_penguin_dt.fit(penguin_X_train, penguin_y_train)
best_penguin_dt = grid_search_penguin_dt.best_estimator_
plot_tree(best_penguin_dt, filled=True, rounded=True, feature_names=penguin_X.columns, class_names=penguin_y.unique())
plt.savefig('penguin-top-dt.png', dpi=400)
plt.close()

grid_search_abalone_dt = GridSearchCV(DecisionTreeClassifier(max_depth=3), param_grid, scoring='accuracy', cv=5)
grid_search_abalone_dt.fit(abalone_X_train, abalone_y_train)
best_abalone_dt = grid_search_abalone_dt.best_estimator_
plot_tree(best_abalone_dt, filled=True, rounded=True, feature_names=abalone_X.columns, class_names=abalone_y.unique())
plt.savefig('abalone-top-dt.png', dpi=400)
plt.close()

penguin_base_mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', random_state=42)
penguin_base_mlp.fit(penguin_X_train, penguin_y_train)

abalone_base_mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', random_state=42)
abalone_base_mlp.fit(abalone_X_train, abalone_y_train)

param_grid_mlp = {
    'activation': ['logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    'solver': ['adam', 'sgd']
}
grid_search_penguin_topmlp = GridSearchCV(MLPClassifier(), param_grid_mlp, scoring='accuracy', cv=5, n_jobs=-1)
grid_search_penguin_topmlp.fit(penguin_X_train, penguin_y_train)
penguin_top_mlp = grid_search_penguin_topmlp.best_estimator_

grid_search_abalone_mlp = GridSearchCV(MLPClassifier(), param_grid_mlp, scoring='accuracy', cv=5, n_jobs=-1)
grid_search_abalone_mlp.fit(abalone_X_train, abalone_y_train)
abalone_top_mlp = grid_search_abalone_mlp.best_estimator_


def save_model_performance(iteration, letter, model, X_test, y_test, dataset_name, model_name, hyperparameters,
                           output_file):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    save_confusion_matrix_plot(conf_matrix, f'{dataset_name} - {model_name}',
                               f'{dataset_name.lower()}-{model_name}-confusion_matrix.png')
    classification_rep = classification_report(y_test, y_pred, target_names=y_test.unique(), output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = classification_rep['macro avg']['f1-score']
    weighted_f1 = classification_rep['weighted avg']['f1-score']
    iteration += 1
    with open(output_file, 'a') as f:
        f.write('-' * 20 + 'Iteration: ' + str(iteration) + ' (' + str(letter) + ') ' + '---- ' + str(
            model_name) + '' + '-' * 20 + '\n')
        f.write(f'{model_name} - {hyperparameters}\n')
        f.write('Confusion Matrix:\n')
        f.write(str(conf_matrix) + '\n')
        f.write('Classification Report:\n')
        f.write(str(classification_rep) + '\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Macro-average F1: {macro_f1}\n')
        f.write(f'Weighted-average F1: {weighted_f1}\n')
        # f.write('-' * 40 + '\n')


penguin_performance_file = 'penguin-performance.txt'
abalone_performance_file = 'abalone-performance.txt'

for i in range(5):
    print(f'******************************* PENGUIN - ITERATION: {i} *******************************\n')
    save_model_performance(i, 'A', penguin_base_dt, penguin_X_test, penguin_y_test, 'Penguin', 'Base-DT', 'Default',
                           penguin_performance_file)
    save_model_performance(i, 'B', best_penguin_dt, penguin_X_test, penguin_y_test, 'Penguin', 'Top-DT',
                           grid_search_penguin_dt.best_params_, penguin_performance_file)
    save_model_performance(i, 'C', penguin_base_mlp, penguin_X_test, penguin_y_test, 'Penguin', 'Base-MLP', 'Default',
                           penguin_performance_file)
    save_model_performance(i, 'D', penguin_top_mlp, penguin_X_test, penguin_y_test, 'Penguin', 'Top-MLP',
                           grid_search_penguin_topmlp.best_params_, penguin_performance_file)

    print(f'******************************* ABALONE - ITERATION: {i} *******************************\n')
    save_model_performance(i, 'A', abalone_base_dt, abalone_X_test, abalone_y_test, 'Abalone', 'Base-DT', 'Default',
                           abalone_performance_file)
    save_model_performance(i, 'B', best_abalone_dt, abalone_X_test, abalone_y_test, 'Abalone', 'Top-DT',
                           grid_search_abalone_dt.best_params_, abalone_performance_file)
    save_model_performance(i, 'C', abalone_base_mlp, abalone_X_test, abalone_y_test, 'Abalone', 'Base-MLP', 'Default',
                           abalone_performance_file)
    save_model_performance(i, 'D', abalone_top_mlp, abalone_X_test, abalone_y_test, 'Abalone', 'Top-MLP',
                           grid_search_abalone_mlp.best_params_, abalone_performance_file)


def calculate_average_performance(dataset_name, model_name, num_iterations, performance_file):
    accuracy_list = []
    macro_f1_list = []
    weighted_f1_list = []

    with open(performance_file, 'r') as f:
        lines = f.read().splitlines()
        for i in range(num_iterations):
            model_info = f'{model_name} - Default' if i == 0 else f'{model_name} - {lines[i * 11 + 1][len(model_name) + 2:]}'

            try:
                accuracy = float(lines[i * 11 + 5][len('Accuracy: '):])
                macro_f1 = float(lines[i * 11 + 6][len('Macro-average F1: '):])
                weighted_f1 = float(lines[i * 11 + 7][len('Weighted-average F1: '):])
            except ValueError:
                accuracy = 0
                macro_f1 = 0
                weighted_f1 = 0
            accuracy_list.append(accuracy)
            macro_f1_list.append(macro_f1)
            weighted_f1_list.append(weighted_f1)

    avg_accuracy = np.mean(accuracy_list)
    avg_macro_f1 = np.mean(macro_f1_list)
    avg_weighted_f1 = np.mean(weighted_f1_list)

    accuracy_var = np.var(accuracy_list)
    macro_f1_var = np.var(macro_f1_list)
    weighted_f1_var = np.var(weighted_f1_list)

    with open(performance_file, 'a') as f:
        f.write(f'Average Accuracy: {avg_accuracy} (Variance: {accuracy_var})\n')
        f.write(f'Average Macro-average F1: {avg_macro_f1} (Variance: {macro_f1_var})\n')
        f.write(f'Average Weighted-average F1: {avg_weighted_f1} (Variance: {weighted_f1_var})\n')


calculate_average_performance('Penguin', 'Base-DT', 5, penguin_performance_file)
calculate_average_performance('Penguin', 'Top-DT', 5, penguin_performance_file)
calculate_average_performance('Penguin', 'Base-MLP', 5, penguin_performance_file)
calculate_average_performance('Penguin', 'Top-MLP', 5, penguin_performance_file)

calculate_average_performance('Abalone', 'Base-DT', 5, abalone_performance_file)
calculate_average_performance('Abalone', 'Top-DT', 5, abalone_performance_file)
calculate_average_performance('Abalone', 'Base-MLP', 5, abalone_performance_file)
calculate_average_performance('Abalone', 'Top-MLP', 5, abalone_performance_file)
