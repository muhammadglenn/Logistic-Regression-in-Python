# Libraries
import pandas as pd
import numpy as np

# Dataset
hearth = pd.read_excel('D:\Datasets\Kaggle\Heart Attack Analysis & Prediction Dataset\heart.xlsx')
print("Hearth Dataset")
print(hearth.head(10))
print("")

# Explore Dataset
print("Info Hearth Dataset")
print(hearth.info())
print("")

# Check Duplicates
print("Number of duplicates")
print(hearth.duplicated().sum())
print("")
print(hearth.loc[hearth.duplicated()])
print("")

# Remove Duplicates
hearth.drop_duplicates(inplace = True)
print("Number of duplicates (check)")
print(hearth.duplicated().sum())
print("")

# Change several numerical variables into category
numtocat = hearth[['gender','cp','fbs','restecg','exang','slp',
                  'caa','thall','output']]
for q in numtocat:
    hearth[q] = hearth[q].astype('category');
print("New type of variables in dataset")
print(hearth.dtypes)
print("")

# Change labels from categorical variables (gender)
print('gender labels before changed')
print(hearth['gender'].unique())
new_gender = pd.Categorical(hearth["gender"])
new_gender = new_gender.rename_categories(["Female","Male"])              
hearth["gender"] = new_gender
print("gender labels after changed")
print(hearth['gender'].unique())
print("")

# Change labels from categorical variables (cp)
print('cp labels before changed')
print(hearth['cp'].unique())
new_cp = pd.Categorical(hearth["cp"])
new_cp = new_cp.rename_categories(["cp_Type1","cp_Type 2","cp_Type 3","cp_Type 4"])              
hearth["cp"] = new_cp
print("cp labels after changed")
print(hearth['cp'].unique())
print("")

# Change labels from categorical variables (fbs)
print('fbs labels before changed')
print(hearth['fbs'].unique())
new_fbs = pd.Categorical(hearth["fbs"])
new_fbs = new_fbs.rename_categories(["fbs_Under120","fbs_Over120"])              
hearth["fbs"] = new_fbs
print("fbs labels after changed")
print(hearth['fbs'].unique())
print("")

# Change labels from categorical variables (restecg)
print('restecg labels before changed')
print(hearth['restecg'].unique())
new_restecg = pd.Categorical(hearth["restecg"])
new_restecg = new_restecg.rename_categories(["restecg_Zero","restecg_One","restecg_Two"])              
hearth["restecg"] = new_restecg
print("restecg labels after changed")
print(hearth['restecg'].unique())
print("")

# Change labels from categorical variables (exang)
print('exang labels before changed')
print(hearth['exang'].unique())
new_exang = pd.Categorical(hearth["exang"])
new_exang = new_exang.rename_categories(["exang_no","exang_yes"])              
hearth["exang"] = new_exang
print("exang labels after changed")
print(hearth['exang'].unique())
print("")

# Change labels from categorical variables (slp)
print('slp labels before changed')
print(hearth['slp'].unique())
new_slp = pd.Categorical(hearth["slp"])
new_slp = new_slp.rename_categories(["slp_Zero","slp_One","slp_Two"])              
hearth["slp"] = new_slp
print("slp labels after changed")
print(hearth['slp'].unique())
print("")

# Change labels from categorical variables (caa)
print('caa labels before changed')
print(hearth['caa'].unique())
new_caa = pd.Categorical(hearth["caa"])
new_caa = new_caa.rename_categories(["caa_Zero","caa_One","caa_Two","caa_Three","caa_Four"])              
hearth["caa"] = new_caa
print("caa labels after changed")
print(hearth['caa'].unique())
print("")

# Change labels from categorical variables (thall)
print('thall labels before changed')
print(hearth['thall'].unique())
new_thall = pd.Categorical(hearth["thall"])
new_thall = new_thall.rename_categories(["thall_Zero","thall_One","thall_Two","thall_Three"])              
hearth["thall"] = new_thall
print("thall labels after changed")
print(hearth['thall'].unique())
print("")

# Change labels from categorical variables (output)
print('output labels before changed')
print(hearth['output'].unique())
new_output = pd.Categorical(hearth["output"])
new_output = new_output.rename_categories(["Output_No","Output_Yes"])              
hearth["output"] = new_output
print("output labels after changed")
print(hearth['output'].unique())
print("")

# Make a list of columns
category = hearth.select_dtypes('category').columns
print("categorical variables")
print(category)
print("")
numeric = hearth.select_dtypes('number').columns
print("numerical variables")
print(numeric)
print("")

# Numerical variables brief
print("Numerical variables brief")
print(hearth.describe())
print("")

# Check how many label on each categorical variabels
for w in category:
    uniquelabel = len(hearth[w].unique())
    print('{namecatvar}  contains {numberlabel} labels'.format(namecatvar=w,numberlabel=uniquelabel));
print('')

# Find outliers for all numerical variables
for e in numeric:
    print('{var} data description'.format(var=e))
    print(round(hearth[e].describe()),2)
    print('')
    IQR = hearth[e].quantile(0.75) - hearth[e].quantile(0.25)
    Lower_fence = hearth[e].quantile(0.25) - (IQR * 3)
    Upper_fence = hearth[e].quantile(0.75) + (IQR * 3)
    print('{outvar} outliers are values < {lowerboundary} or > {upperboundary}'.format(outvar=e,lowerboundary=Lower_fence, upperboundary=Upper_fence))
    print('');

# Handling outliers
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [hearth]:
    df3['chol'] = max_value(df3, 'chol', 466);

# Declare x and y
X = hearth.drop(['output'], axis=1)
y = hearth['output']
print('x datatype')
print(X.dtypes)
print('')

# Split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Check the shape of X_train and X_test
print('x train shape')
print(X_train.shape)
print('x test shape')
print(X_test.shape)
print('')

# Make dummies variables from categorical variables
Xtrain_num = X_train.select_dtypes('number').columns
X_train = pd.concat([X_train[Xtrain_num], 
                    pd.get_dummies(X_train.gender), 
                    pd.get_dummies(X_train.cp),
                    pd.get_dummies(X_train.fbs),
                    pd.get_dummies(X_train.restecg),
                    pd.get_dummies(X_train.exang),
                    pd.get_dummies(X_train.slp),
                    pd.get_dummies(X_train.caa),
                    pd.get_dummies(X_train.thall)], axis=1);

Xtest_num = X_test.select_dtypes('number').columns
X_test = pd.concat([X_test[Xtrain_num], 
                    pd.get_dummies(X_test.gender), 
                    pd.get_dummies(X_test.cp),
                    pd.get_dummies(X_test.fbs),
                    pd.get_dummies(X_test.restecg),
                    pd.get_dummies(X_test.exang),
                    pd.get_dummies(X_test.slp),
                    pd.get_dummies(X_test.caa),
                    pd.get_dummies(X_test.thall)], axis=1)

#Check columns after making dummies variables
print('X_train columns')
for colx in X_train.columns:
    print(colx);
print('')
print('X_test columns')
for colx in X_test.columns:
    print(colx);
print('')

# Feature Scaling
print('X_train description before scaling')
print(X_train.describe())
print('')

cols = X_train.columns

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

print(' X train description after scaling')
print(X_train.describe())
print('')

# Model Training
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)
# fit the model
logreg.fit(X_train, y_train)

# Predict result
y_pred_test = logreg.predict(X_test)
print(y_pred_test)
print('')

# Check accuracy score 
from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
print('')

# Check for overfitting and underfitting
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
print('')

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# Precision score
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))

# Recall or Sensitivity
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))

# Specificity
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))