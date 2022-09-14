from sklearn import neighbors, naive_bayes, neural_network, svm, tree
from sklearn.metrics import confusion_matrix,  roc_curve, auc, matthews_corrcoef, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

#'data/pd_speech_features.txt'
df = pd.read_csv('./pd_speech_features.txt')


# Correlation matrix
df.dataframeName = 'pd_speech_features.csv'
 
X = np.array(df.drop(['class'], 1)) #matrix 755*755
y = np.array(df['class']) #solution variable
#normalization
Z = np.divide((X - X.mean(0)), X.std(0))

pca = PCA(n_components = 168) 
Z_PCA = pca.fit_transform(Z)

# KNN
# knn1 = neighbors.KNeighborsClassifier(n_neighbors=7)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# Naive Bayes
gnb = naive_bayes.GaussianNB()

#Random Forest with two Hyperparameters
rf = RandomForestClassifier(max_depth=2, random_state=42)

# Support Vector Machine
svmc = svm.SVC(kernel='linear', probability=True,
                     random_state=1)

# Neural Network
nn = neural_network.MLPClassifier(
                                   hidden_layer_sizes = (50, 50, 20), #will reduce the layers (500,500,200)
                                   random_state = 42)
# Decision Tree
dt = tree.DecisionTreeClassifier(random_state = 42)

# XGBoost
xgboost = xgb.XGBClassifier(random_state=1,learning_rate=0.01)


folds = 5
cv = StratifiedKFold(n_splits=folds)

# Classifiers:
# Decision Tree: dt
# MLP: nn
# Random Forest: rf
# KNN: knn
# XGBoost: xgboost
#naive-bayes: gnb
#svm : svmc

classifier = xgboost
inp = Z

acc = np.zeros(folds)
confm = np.zeros((2, 2))
mcc = np.zeros(folds)
f1 = np.zeros(folds)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(inp, y):
    probas = classifier.fit(inp[train], y[train]).predict_proba(inp[test])
    # Compute accuracy
    y_pred = classifier.predict(inp[test])
    acc[i] = (y_pred == y[test]).mean()
    # Compute MCC
    mcc[i] = matthews_corrcoef(y[test], y_pred)
    # Compute f1
    f1[i] = f1_score(y[test], y_pred)
    # Confusion matrix
    confm = confm + confusion_matrix(y[test], y_pred)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    #begin from 0,0
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # linewidth = 1, alpha,
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print("Accuracy: ")
print('{:.2f}% +- {:.2f}%'.format(acc.mean() * 100, acc.std() * 100))
cm = np.zeros((3,3))
cm[0:2, 0:2] = confm
cm[0,2] = (cm[0,0] / cm[0,0:2].sum())* 100
cm[1,2] = (cm[1,1] / cm[1,0:2].sum())* 100
cm[2,0] = (cm[0,0] / cm[0:2,0].sum())* 100
cm[2,1] = (cm[1,1] / cm[0:2,1].sum())* 100
print("Confusion Matrix:")
print(cm)
print("Matthews correlation coefficient (MCC): ")
print('{:.2f}%'.format(mcc.mean() * 100))
print("F1 score: ")
print('{:.2f}%'.format(f1.mean() * 100))
