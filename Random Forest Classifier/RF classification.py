# -*- coding: utf-8 -*-
"""
Authors:

Zakiah Darajat Nurfajrin
        Doctoral Student
        Department of Applied Chemistry, Graduate School of Engineering, Kyushu University
        744 Motooka, Nishi-ku, Fukuoka 819-0395, Japan
        Email: nurfajrin.zakiah.225@s.kyushu-u.ac.jp / zakiahdarajat.zdn@gmail.com

Selectivity of ILs toward particular metals
Learning and prediction by Random Forest Classifier
Data abbreviations:
    Ni  = Nickel
    Co  = Cobalt
    Mn  = Manganese
    Li  = Lithium

"""

# Load the packages
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker



# Load and define dataframe for the learning dataset
Learning = pd.read_csv("C:/Users/GKlab/Documents/PENELITIAN ZAKIAH/1st PAPER/ML/UPLOAD TO GIT/XGB/descriptor for conventional metal extractant.csv")

print('\t')
print('Learning dataset (original): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)
print(Learning.describe())

# Convert non-numeric data to numeric
Learning.loc[Learning.metal == 'Ni', 'metal'] = 0
Learning.loc[Learning.metal == 'Co', 'metal'] = 1
Learning.loc[Learning.metal == 'Li', 'metal'] = 2
Learning.loc[Learning.metal == 'Mn', 'metal'] = 3

print('\n')
print('Learning dataset (converted): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)

# Define X and y out of the learning data (X: features, y: label)
X = Learning.drop('metal', axis=1)
y = Learning['metal'].values
y = y.astype('int')

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)

# Define the parameter space for hyperparameters
param_space = {
    'learning_rate': [0.001, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'lambda': [0, 1, 10]
}

# Fit the model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# XGBoost (different learning rate)
learning_rate_range = np.arange(0.01, 1, 0.5)
test_XG = [] 
train_XG = []
for lr in learning_rate_range:
    xgb_classifier = xgb.XGBClassifier(eta = lr)
    xgb_classifier.fit(X_train, y_train)
    train_XG.append(xgb_classifier.score(X_train, y_train))
    test_XG.append(xgb_classifier.score(X_test, y_test))

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

#XGBoost performs
fig = plt.figure(figsize=(10, 7))
plt.plot(learning_rate_range, train_XG, c='orange', label='Train')
plt.plot(learning_rate_range, test_XG, c='m', label='Test')
plt.xlabel('Learning rate')
plt.xticks(learning_rate_range)
plt.ylabel('Accuracy score')
plt.ylim(0.6, 1)
plt.legend(prop={'size': 12}, loc=3)
plt.title('Accuracy score vs. Learning rate of XGBoost', size=14)
plt.show()

# Create an instance of RandomizedSearchCV
random_search = RandomizedSearchCV(
    xgb_classifier,
    param_distributions=param_space,
    n_iter=50,  # Number of random combinations to try
    scoring='accuracy',
    cv=5,  # Number of cross-validation folds
    n_jobs=-1,  # Number of parallel jobs (use -1 to utilize all available cores)
    random_state=5
)

# Fit the RandomizedSearchCV instance on training data
random_search.fit(X_train, y_train)

# Access the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the model using the best hyperparameters
best_model = random_search.best_estimator_
accuracy_train = best_model.score(X_train, y_train)
accuracy_test = best_model.score(X_test, y_test)
print("Training Accuracy:", accuracy_train)
print("Test Accuracy:", accuracy_test)


### Analyze and visualize the optimized model performance on TRAINING SET using CROSS-VALIDATION
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict

# Cross-Validation Evaluation on Training Set
cv_pred = cross_val_predict(best_model, X_train, y_train, cv=5)


print('Quality assessment with cross-validation:')
print('Accuracy score:', accuracy_score(y_train, cv_pred))
print('Classification report:\n', classification_report(y_train, cv_pred))
print('Confusion matrix:\n', confusion_matrix(y_train, cv_pred))



# Plot Confusion Matrix using ConfusionMatrixDisplay
fonts = fm.FontProperties(family='arial', size=20, weight='normal', style='normal')
categories = ['Ni', 'Mn', 'Co', 'Li']
cm_cv = confusion_matrix(y_train, cv_pred, labels=model.classes_)
disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=categories)
fig, ax = plt.subplots(figsize=(5, 5))
disp_cv.plot(ax=ax, cmap='Blues')
plt.xlabel('Predicted Label', labelpad=10, fontproperties=fm.FontProperties(family='arial', size=20, weight='normal', style='normal'))
plt.ylabel('True Label', labelpad=10, fontproperties=fm.FontProperties(family='arial', size=20, weight='normal', style='normal'))
dpi_assign = 300
plt.savefig('fig1a.jpg', dpi=dpi_assign, bbox_inches='tight')
#### Analyze and visualize the optimized model performance on TRAINING SET via a SINGLE RUN
train_pred = model.predict(X_train)

print('\n')
print('Learning results for training set (employ model with the best hyperparameters): \n')
print('Accuracy score: ', accuracy_score(y_train, train_pred))
print('Classification report: \n', classification_report(y_train, train_pred))
print('Confusion matrix: \n', confusion_matrix(y_train, train_pred))

### Plot using ConfusionMatrixDisplay
cm_train = confusion_matrix(y_train, train_pred, labels=model.classes_)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=categories)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
disp_train.plot(ax=ax, cmap='Greys')
plt.xlabel('Predicted Label', labelpad=10, fontproperties=fonts)
plt.ylabel('True Label', labelpad=10, fontproperties=fonts)
dpi_assign = 300
plt.savefig('figs1a.jpg', dpi=dpi_assign, bbox_inches='tight')

### Analyze and visualize the optimized model performance on TEST SET via a SINGLE RUN
test_pred = model.predict(X_test)

print('\n')
print('Learning results for test set (employ model with the best hyperparameters): \n')
print('Accuracy score: ', accuracy_score(y_test, test_pred))
print('Classification report: \n', classification_report(y_test, test_pred))
print('Confusion matrix: \n', confusion_matrix(y_test, test_pred))

### Plot using ConfusionMatrixDisplay
cm_test = confusion_matrix(y_test, test_pred, labels=model.classes_)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=categories)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
disp_test.plot(ax=ax, cmap='Reds')
plt.xlabel('Predicted Label', labelpad=10, fontproperties=fonts)
plt.ylabel('True Label', labelpad=10, fontproperties=fonts)
dpi_assign = 300
plt.savefig('fig1b.jpg', dpi=dpi_assign, bbox_inches='tight')

# Visualize Feature Importances
feature_importances = pd.DataFrame({'features': X_train.columns, 'importance': model.feature_importances_})
fig = plt.figure(figsize=(12,5))
ax = sns.barplot(x='features', y='importance', data=feature_importances, palette='vlag')
plt.xlabel('Feature', labelpad=20, fontproperties=fm.FontProperties(family='arial', size=10, weight='normal', style='normal'))
plt.ylabel('Importance', labelpad=20, fontproperties=fm.FontProperties(family='arial', size=10, weight='normal', style='normal'))
ticker_arg = [0.025, 0.05, 0.025, 0.05]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.yaxis.set_minor_locator(tickers[0])
ax.yaxis.set_major_locator(tickers[1])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(10)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(10)) for j in ycoord]
dpi_assign = 300
plt.savefig('fig2a.jpg', dpi=dpi_assign, bbox_inches='tight')

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np

# Use label_binarize to convert the multiclass target variable to binary
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])  

# Define the classifier using OneVsRestClassifier
classifier = make_pipeline(StandardScaler(), OneVsRestClassifier(LinearSVC(random_state=10)))

# Fit the classifier
classifier.fit(X_train, y_train)

# Compute decision function for each class
y_score = classifier.decision_function(X_test)

# Determine the number of unique classes in the target variable
n_classes = len(np.unique(y_train))

# Compute precision-recall curve and average precision for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# Define colors for each class
colors = ['orange', 'green', 'red', 'purple']

# Plot the precision-recall curve for each class
plt.figure(figsize=(8, 6))
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'Precision-recall for class {i} (AP={average_precision[i]:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# plt.title('Precision-Recall curve for each class')
plt.legend(loc="best")

plt.savefig('RF - precision-recall curve for each class.png', dpi=1300)
plt.show()


##======================================================================================================##
### Load descriptors for actual predictions
descriptors = pd.read_csv("C:/Users/GKlab/Documents/PENELITIAN ZAKIAH/1st PAPER/ML/XGB/DATA TEST_DES.csv") # This contains descriptors (features) for 150 chemicals

print('\n')
print('Descriptor data: ')
print(f'Filetype: {type(descriptors)}, Shape: {descriptors.shape}')
print(descriptors)
print(descriptors.describe())

### Predict the class of each descriptor i.e. metal selectivity
label_pred = model.predict(descriptors)

print('\n')
print('Prediction of descriptor data: ')
print(label_pred)

Prediction = pd.DataFrame(label_pred, columns = ['Selectivity']) # Covert numpy to pandas
Prediction.Selectivity[Prediction.Selectivity == 0] = 'Ni'
Prediction.Selectivity[Prediction.Selectivity == 1] = 'Co'
Prediction.Selectivity[Prediction.Selectivity == 2] = 'Li'
Prediction.Selectivity[Prediction.Selectivity == 3] = 'Mn'

print('\n')
print('Prediction of descriptor data (converted): ')
print(Prediction)
print(Prediction.value_counts())

# Save the predictions to a CSV file
prediction_filename = "prediction_results.csv"
Prediction.to_csv(prediction_filename, index=False)

print('\n')
print(f'Prediction results saved to {prediction_filename}')
