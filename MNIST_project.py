import numpy as np
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
np.random.seed(42)
PROJECT_ROOT_DIRECTORY = '.'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'images')

import matplotlib.pyplot as plt

def save_figure(figure_id, tight_layout = False, figure_extension = 'png', resolution = 300):
    path = os.path.join(IMAGES_PATH, figure_id + '.' + figure_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = figure_extension, dpi = resolution)
    
from sklearn.datasets import fetch_openml

def sort_by_target(mnist):
    reorder_train_set = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test_set = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train_set]
    mnist.target[:60000] = mnist.target[reorder_train_set]
    mnist.data[60000:] = mnist.data[reorder_test_set + 60000]
    mnist.target[60000:] = mnist.target[reorder_test_set + 60000]
    
mnist = fetch_openml('mnist_784', version = 1, cache = True, as_frame = False, parser = 'auto')
mnist.target = mnist.target.astype(np.int8)
sort_by_target(mnist)
##print(mnist['data'], mnist['target'])
X, y = mnist['data'], mnist['target']
##print(X.shape, y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation = 'nearest')
plt.axis('off')
save_figure('example_03_some_digit_image_03')
##plt.show()
plt.close()
##print(y[36000])

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary, interpolation = 'nearest')
    plt.axis('off')
    
def plot_digits(instances, images_per_row = 10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    number_of_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    number_of_empty = number_of_rows * len(instances)
    images.append(np.zeros((size, size * number_of_empty)))
    for row in range(number_of_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis('off')

example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]    
plt.figure(figsize = (9, 9))
plot_digits(example_images, images_per_row = 10)
save_figure('example_03_digits')
##plt.show()
plt.close()

X_train_set, X_test_set, y_train_set, y_test_set = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train_set, y_train_set = X[shuffle_index], y[shuffle_index]

y_train_for_5 = (y_train_set == 5)
y_test_for_5 = (y_test_set == 5)

from sklearn.linear_model import SGDClassifier

sgd_classifier = SGDClassifier(max_iter = 5, random_state = 42)
sgd_classifier.fit(X_train_set, y_train_for_5)
##print(sgd_classifier.predict([some_digit]))

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits = 3, random_state = 42, shuffle = True)

for train_index, test_index in skfolds.split(X_train_set, y_train_for_5):
    clone_classifier = clone(sgd_classifier)
    X_train_folds = X_train_set[train_index]
    y_train_folds = y_train_for_5[train_index]
    X_test_folds = X_train_set[test_index]
    y_test_folds = y_train_for_5[test_index]
    clone_classifier.fit(X_train_folds, y_train_folds)
    y_prediction = clone_classifier.predict(X_test_folds)
    number_of_correct = sum(y_prediction == y_test_folds)
    ##print(number_of_correct / len(y_prediction))
    

from sklearn.model_selection import cross_val_score
##print(cross_val_score(sgd_classifier, X_train_set, y_train_for_5, cv = 3, scoring = 'accuracy'))


'''

10.07.2023
Continue from page 106  

'''


from sklearn.base import BaseEstimator

class NeverFiveClassifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype = bool)
    
never_five_classifier = NeverFiveClassifier()
##print(cross_val_score(never_five_classifier, X_train_set, y_train_for_5, cv = 3, scoring = 'accuracy'))

from  sklearn.model_selection import cross_val_predict

y_train_prediction = cross_val_predict(sgd_classifier, X_train_set, y_train_for_5, cv = 5)

from sklearn.metrics import confusion_matrix

##print(confusion_matrix(y_train_for_5, y_train_prediction))
y_train_perfect_prediction = y_train_for_5
##print(confusion_matrix(y_train_for_5, y_train_perfect_prediction))

from sklearn.metrics import precision_score, recall_score, f1_score

print()
print('SGD at 0 Threshold')
print('Precision:', precision_score(y_train_for_5, y_train_prediction))
# or (True Positive / (True Positive + False Positive))
print('Recall:', recall_score(y_train_for_5, y_train_prediction))
# or (True Positive / (True Positive + True Negative))
print('F1 Score:', f1_score(y_train_for_5, y_train_prediction))

y_scores = sgd_classifier.decision_function([some_digit])
##print('y scores: ', y_scores)
threshold = 0
y_some_digit_prediction = (y_scores > threshold)
##print('y_some_digit_prediction: ', y_some_digit_prediction)
threshold = 200000
y_some_digit_prediction = (y_scores > threshold)
##print('y_some_digit_prediction: ', y_some_digit_prediction)

# Optimal Threshold
y_scores = cross_val_predict(sgd_classifier, X_train_set, y_train_for_5, cv = 3, method = 'decision_function')
if y_scores.ndim == 2:
    y_scores = y_scores[:, 1]

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_for_5, y_scores)

def plot_precision_recall_vs_threshold (precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b-.', label = 'Precision', linewidth = 2)
    plt.plot(thresholds, recalls[:-1], 'g-', label = 'Recall', linewidth = 2)
    plt.xlabel('Threshold', fontsize = 16)
    plt.legend(loc = 'upper left', fontsize = 16)
    plt.ylim([0, 1])
    plt.xlim([-700000, 700000])
    
plt.figure(figsize = (8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
save_figure('example_03_precision_recall_vs_threshold_plot')
##plt.show()
plt.close()

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b-', linewidth = 2)
    plt.xlabel('Recall', fontsize = 16)
    plt.ylabel('Precision', fontsize = 16)
    plt.axis([0, 1, 0, 1])
    
plt.figure(figsize = (8, 6))
plot_precision_vs_recall(precisions, recalls)
save_figure('example_03_precision_vs_recall_plot')
##plt.show()
plt.close()

# Threshold for my case should be about -3000 for ∼87% Precision, ∼64% Recall for 100 iteration.
# Threshold for my case should be about 70000 for ∼87% Precision, ∼65% Recall for 5 iteration.

y_train_prediction_90 = (y_scores > 70000)
print()
print('SGD at 70000 Threshold')
print('Precision is: ', precision_score(y_train_for_5, y_train_prediction_90))
print('Recall is: ', recall_score(y_train_for_5, y_train_prediction_90))

from sklearn.metrics import roc_curve

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_for_5, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label = None):
    plt.plot(false_positive_rate, true_positive_rate, 'r-', linewidth = 2, label = label)
    plt.plot([0, 1], [0, 1], 'k--', linewidth = 2)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)
    
plt.figure(figsize = (8, 6))
plot_roc_curve(false_positive_rate, true_positive_rate)
save_figure('example_03_ROC_curve')
##plt.show()
plt.close()

from sklearn.metrics import roc_auc_score

print('ROC/AUC Score is:', roc_auc_score(y_train_for_5, y_scores))

from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators = 10, random_state = 42)
y_probabilities_forest = cross_val_predict(forest_classifier, X_train_set, y_train_for_5, cv = 3, method = 'predict_proba')
y_scores_forest = y_probabilities_forest[:, 1] # All positive
false_positive_rate_forest, true_positive_rate_forest, thresholds_forest = roc_curve(y_train_for_5, y_scores_forest)

plt.figure(figsize = (8, 6))
plt.plot(false_positive_rate, true_positive_rate, 'b:', linewidth = 2, label = 'SGD')
plot_roc_curve(false_positive_rate_forest, true_positive_rate_forest, 'Random Forest')
plt.legend(loc = 'lower right', fontsize = 16)
save_figure('example_03_forest_vs_sgd_plot')
##plt.show()
plt.close()

y_train_prediction_forest = cross_val_predict(forest_classifier, X_train_set, y_train_for_5, cv = 3)
print()
print('Random Forest')
print('Precision is:', precision_score(y_train_for_5, y_train_prediction_forest))
print('Recall is:', recall_score(y_train_for_5, y_train_prediction_forest))
print('ROC/AUC Score is:', roc_auc_score(y_train_for_5, y_scores_forest))

# Multi Class Training
sgd_classifier.fit(X_train_set, y_train_set)
print()
print('Multi Class SGD')
print('5 Prediction:', sgd_classifier.predict([some_digit]))
some_digit_scores = sgd_classifier.decision_function([some_digit])
print('All predictions:')
print(some_digit_scores)

forest_classifier.fit(X_train_set, y_train_set)
print()
print('Multi Class Random Forest')
print('5 Prediction:', forest_classifier.predict([some_digit]))
print('Probabilites:')
print(forest_classifier.predict_proba([some_digit]))

sgd_multi_class_accuracy = cross_val_score(sgd_classifier, X_train_set, y_train_set, cv = 3, scoring = 'accuracy')
print()
print('SGD Multi Class Accuracy is:', (sum(sgd_multi_class_accuracy) / len(sgd_multi_class_accuracy)))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_set.astype(np.float64))

sgd_multi_class_accuracy_scaled = cross_val_score(sgd_classifier, X_train_scaled, y_train_set, cv = 3, scoring = 'accuracy')
print()
print('SGD Multi Class Accuracy After Scaling is:', (sum(sgd_multi_class_accuracy_scaled) / len(sgd_multi_class_accuracy_scaled)))

y_train_prediction = cross_val_predict(sgd_classifier, X_train_scaled, y_train_set, cv = 3)
confusion_matrix_variable = confusion_matrix(y_train_set, y_train_prediction)
print()
print('Confusion Matrix:')
print(confusion_matrix_variable)

def plot_confusion_matrix(matrix):
    figure = plt.figure(figsize = (8, 8))
    ax = figure.add_subplot(111)
    cax = ax.matshow(matrix)
    figure.colorbar(cax)

plot_confusion_matrix(confusion_matrix_variable)   
save_figure('example_03_confusion_matrix_plot')
##plt.show()
plt.close()

row_sums = confusion_matrix_variable.sum(axis = 1, keepdims = True)
normal_confusion_matrix = confusion_matrix_variable / row_sums
np.fill_diagonal(normal_confusion_matrix, 0)
plot_confusion_matrix(normal_confusion_matrix)
save_figure('example_03_normal_confusion_matrix_plot')
##plt.show()
plt.close()

cl_a, cl_b = 3, 5
X_aa = X_train_set[(y_train_set == cl_a) & (y_train_prediction == cl_a)]
X_ab = X_train_set[(y_train_set == cl_a) & (y_train_prediction == cl_b)]
X_ba = X_train_set[(y_train_set == cl_b) & (y_train_prediction == cl_a)]
X_bb = X_train_set[(y_train_set == cl_b) & (y_train_prediction == cl_b)]

plt.figure(figsize = (8, 8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row = 5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row = 5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row = 5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row = 5)
save_figure('example_03_error_analysis_digits_plot')
##plt.show()
plt.close()


'''

11.07.2023
Will end here.
TODO for 12.07.2023
Add plot_digits() at line 53
Continue from page 121.

12.07.2023
END TODO of 11.07.2023

'''


from sklearn.neighbors import KNeighborsClassifier

y_train_for_large = (y_train_set >= 7)
y_train_for_odd = (y_train_set % 2 == 1)
y_multilabel = np.c_[y_train_for_large, y_train_for_odd]
k_neighbor_classifier = KNeighborsClassifier()
k_neighbor_classifier.fit(X_train_set, y_multilabel)
print()
print('K Neighbour Prediction for Multilabel with Digit 5:')
print('First Boolean is for >= to 7, Second Boolean is for Odd Digit.')
print(k_neighbor_classifier.predict([some_digit]))

y_train_set_k_neighbor_classifier_prediction = cross_val_predict(k_neighbor_classifier, X_train_set, y_multilabel, cv = 3, n_jobs = 1)
print()
print('Average F1 Score of every label is:', f1_score(y_multilabel, y_train_set_k_neighbor_classifier_prediction, average = 'macro'))

noise = np.random.randint(0, 100, (len(X_train_set), 784))
X_train_set_mod = X_train_set + noise
noise = np.random.randint(0, 100, (len(X_test_set), 784))
X_test_set_mod = X_test_set + noise
y_train_set_mod = X_train_set
y_test_set_mod = X_test_set

some_index = 5500
plt.subplot(121); plot_digit(X_test_set_mod[some_index])
plt.subplot(122); plot_digit(y_test_set_mod[some_index])
save_figure('example_03_noisy_vs_clear_plot')
##plt.show()
plt.close()

k_neighbor_classifier.fit(X_train_set_mod, y_train_set_mod)
clean_digit = k_neighbor_classifier.predict([X_test_set_mod[some_index]])
plot_digit(clean_digit)
save_figure('example_03_cleared_noise_digit_plot')
##plt.show()
plt.close()


'''

MNIST Example Done.
Classification Example Done.
Precision/Recall Understood.
Will check exercises of Chapter 2
Page 124
12.07.2023

'''