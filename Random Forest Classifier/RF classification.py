# -*- coding: utf-8 -*-
"""

Authors:

    (1) Adroit T.N. Fajar, Ph.D.
        JSPS Postdoctoral Fellow | 日本学術振興会外国人特別研究員
        Department of Applied Chemistry, Graduate School of Engineering, Kyushu University
        744 Motooka, Nishi-ku, Fukuoka 819-0395, Japan
        Email: fajar.toriq.nur.adroit.009@m.kyushu-u.ac.jp / adroit.fajar@gmail.com
        Scopus Author ID: 57192386143
        Google Scholar: https://scholar.google.com/citations?user=o6jQEEMAAAAJ&hl=en&oi=ao
        ResearchGate: https://www.researchgate.net/profile/Adroit-Fajar
        
    (2) Aditya Dewanto Hartono, Ph.D.
        Postdoctoral Fellow
        Mathematical Modeling Laboratory
        Center for Promotion of International Education and Research
        Department of Agro-environmental Sciences, Faculty of Agriculture, Kyushu University
        744 Motooka, Nishi-ku, Fukuoka 819-0395, Japan
        Email: adityadewanto@gmail.com
        ResearchGate: https://www.researchgate.net/profile/Aditya-Hartono
        
    (3) Zakiah Darajat Nurfajrin
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


### Configure the number of available CPU
import os as os
cpu_number = os.cpu_count()
n_jobs = cpu_number - 2

### Import some standard libraries
import pandas as pd
import seaborn as sns
# import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

### Load and define dataframe for the learning dataset
Learning = pd.read_csv("C:/Users/GKlab/Documents/PENELITIAN ZAKIAH/1st PAPER/ML/RF/descriptor for conventional metal extractant.csv") 

print('\t')
print('Learning dataset (original): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)
print(Learning.describe())

### Convert non-numeric data to numeric

Learning.metal[Learning.metal == 'Ni'] = 1
Learning.metal[Learning.metal == 'Co'] = 2
Learning.metal[Learning.metal == 'Li'] = 3
Learning.metal[Learning.metal == 'Mn'] = 4


print('\n')
print('Learning dataset (converted): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)

# Define X and Y out of the learning data (X: features, Y: label)
X = Learning.drop('metal', axis=1)
Y = Learning['metal'].values
Y = Y.astype('int')


### Split the learning data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

cross_val = 10

### Train and evaluate the model
from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(random_state=10)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RFclf, X_train, Y_train, scoring="accuracy", cv=cross_val) 

def display_score(scores):
    print('\n')
    print('Preliminary run: \n')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_score(scores)

### Fine tune the model using RandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
param_space = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [100, 200, 300, 400, 500],
    'max_features': [2, 4, 6, 8, 10, 12]
}
random_search = RandomizedSearchCV(
    RFclf,
    param_distributions=param_space,
    n_iter=10,  # Number of iterations for random sampling
    scoring="accuracy",
    cv=cross_val,
    n_jobs=n_jobs
)
random_search.fit(X_train, Y_train)

best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

cvres = random_search.cv_results_

print('\n')
print('Hyperparameter tuning: \n')
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

best_estimator = random_search.best_estimator_

import matplotlib.pyplot as plt
import seaborn as sns

# Extract relevant information from cvres and best model
mean_test_scores = cvres['mean_test_score']
best_test_score = random_search.best_score_
param_combinations = [str(param) for param in random_search.cv_results_['params']]

# Set custom color palette
custom_palette = sns.color_palette("Set2")

# Plot results for different hyperparameter settings
plt.figure(figsize=(10, 6))

# Customize line plot
sns.lineplot(x=param_combinations, y=mean_test_scores, marker='o', label='Cross-Validation', color=custom_palette[0])
plt.axhline(y=best_test_score, color=custom_palette[1], linestyle='--', label='Best Test Score')

# Customize labels and title
plt.xlabel('Hyperparameter Combination')
plt.ylabel('Mean Test Score')
plt.title('Cross-Validation Results vs. Best Test Score')

# Customize legend
plt.legend()

# Customize x-axis ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Set grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot with high DPI
plt.savefig('customized_plot.png', dpi=300)

# Show the plot
plt.show()


### Re-train the model with the best hyperparameters and the whole training set
RFclf_opt = random_search.best_estimator_
model = RFclf_opt.fit(X_train, Y_train)

### Analyze and visualize the optimized model performance on TRAINING SET using CROSS-VALIDATION
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict

cv_pred = cross_val_predict(RFclf_opt, X_train, Y_train, cv=cross_val)

print('\n')
print('Quality assessment with cross-validation (employ model with the best hyperparameters): \n')
print('Accuracy score: ', accuracy_score(Y_train, cv_pred)) ## ini nilai yang paling penting (> 0.7)
print('Classification report: \n', classification_report(Y_train, cv_pred))
print('Confusion matrix: \n', confusion_matrix(Y_train, cv_pred))


import matplotlib.pyplot as plt
import seaborn as sns

# Extract relevant information from cvres and best model
mean_test_scores = cvres['mean_test_score']
best_test_score = random_search.best_score_

# Set custom color palette
custom_palette = sns.color_palette("Set2")

# Plot results for different hyperparameter settings
plt.figure(figsize=(10, 6))

# Customize line plot
sns.lineplot(x=range(len(mean_test_scores)), y=mean_test_scores, marker='o', label='Cross-Validation', color=custom_palette[0])
plt.axhline(y=best_test_score, color=custom_palette[1], linestyle='--', label='Best Test Score')

# Customize labels and title
plt.xlabel('Hyperparameter Combination')
plt.ylabel('Mean Test Score')
plt.title('Cross-Validation Results vs. Best Test Score')

# Customize legend
plt.legend()

# Customize x-axis ticks
plt.xticks(range(len(mean_test_scores)), [str(param) for param in random_search.cv_results_['params']], rotation=45, ha='right')
plt.tight_layout()

# Set grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot with high DPI
plt.savefig('customized_plot.png', dpi=300)

# Show the plot
plt.show()


### Plot using ConfusionMatrixDisplay
import matplotlib.font_manager as fm
fonts = fm.FontProperties(family='arial', size=20, weight='normal', style='normal')
categories = 'Ni', 'Co', 'Mn', 'Li'
cm_cv = confusion_matrix(Y_train, cv_pred, labels=RFclf_opt.classes_)
disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=categories)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
disp_cv.plot(ax=ax, cmap='Blues')
plt.xlabel('Predicted Label', labelpad=10, fontproperties=fonts)
plt.ylabel('True Label', labelpad=10, fontproperties=fonts)
dpi_assign = 300
plt.savefig('fig1a.jpg', dpi=dpi_assign, bbox_inches='tight')

#### Analyze and visualize the optimized model performance on TRAINING SET via a SINGLE RUN
train_pred = model.predict(X_train)

print('\n')
print('Learning results for training set (employ model with the best hyperparameters): \n')
print('Accuracy score: ', accuracy_score(Y_train, train_pred))
print('Classification report: \n', classification_report(Y_train, train_pred))
print('Confusion matrix: \n', confusion_matrix(Y_train, train_pred))

### Plot using ConfusionMatrixDisplay
cm_train = confusion_matrix(Y_train, train_pred, labels=RFclf_opt.classes_)
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
print('Accuracy score: ', accuracy_score(Y_test, test_pred)) 
print('Classification report: \n', classification_report(Y_test, test_pred))
print('Confusion matrix: \n', confusion_matrix(Y_test, test_pred))

### Plot using ConfusionMatrixDisplay
cm_test = confusion_matrix(Y_test, test_pred, labels=RFclf_opt.classes_)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=categories)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
disp_test.plot(ax=ax, cmap='Reds')
plt.xlabel('Predicted Label', labelpad=10, fontproperties=fonts)
plt.ylabel('True Label', labelpad=10, fontproperties=fonts)
dpi_assign = 300
plt.savefig('fig1b.jpg', dpi=dpi_assign, bbox_inches='tight')

# Fit the model with the best hyperparameters
RFclf_opt = random_search.best_estimator_
model = RFclf_opt.fit(X_train, Y_train)

# Extract and visualize feature importances
feature_importances = pd.DataFrame({'features': X_train.columns, 'importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Print feature importances
print('\nFeature Importances:\n')
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(12, 5))
ax = sns.barplot(x='features', y='importance', data=feature_importances, palette='vlag')
plt.xlabel('Feature', labelpad=10, fontproperties=fonts)
plt.ylabel('Importance', labelpad=10, fontproperties=fonts)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('feature_importance_random_forest.png', dpi=300)
plt.show()


### Load descriptors for actual predictions
descriptors = pd.read_csv("C:/Users/GKlab/Documents/PENELITIAN ZAKIAH/1st PAPER/ML/RF/DATA TEST_DES.csv") 

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
Prediction.Selectivity[Prediction.Selectivity == 1] = 'Ni'
Prediction.Selectivity[Prediction.Selectivity == 2] = 'Co'
Prediction.Selectivity[Prediction.Selectivity == 3] = 'Li'
Prediction.Selectivity[Prediction.Selectivity == 4] = 'Mn'

print('\n')
print('Prediction of descriptor data (converted): ')
print(Prediction)
print(Prediction.value_counts())

# Save the predictions to a CSV file
prediction_filename = "prediction_results.csv"
Prediction.to_csv(prediction_filename, index=False)

print('\n')
print(f'Prediction results saved to {prediction_filename}')
