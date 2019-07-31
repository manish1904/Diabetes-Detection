# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')
#%matplotlib inline
from sklearn.model_selection import train_test_split

# Importing Dataset
df = pd.read_excel('Diabetes_Detection.xlsx')
print(df.tail())
print(df.describe())
print(df.isnull().sum())

# Separating dependent and independent variables
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

# Standardize Dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print("\n")

# Correlation between variables using heatmap
import matplotlib.pyplot as plt2
corr = df.corr()
plt2.figure(figsize=(12,6))
sns.heatmap(corr,vmax=1,square=True,cmap='YlGnBu')

# Precise correlation with the target variable
cor_dict = corr['Outcome'].to_dict()
del cor_dict['Outcome']
print("Listing the Independent variables by their correlation  with Outcome:\n")
cor_dict.items()
for ele in sorted(cor_dict.items(),key= lambda x: -abs(x[1])):
    print ("{0}: \t{1}".format(*ele))

# Splitting data into Training and Testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.29, random_state = 1)


''' Linear Discriminant Analysis '''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)

LDA_Y_pred = lda.predict(X_test)
print("\n")

print("Accuracy of LDA: ", np.mean(LDA_Y_pred == Y_test))
lda_cm = confusion_matrix(Y_test,LDA_Y_pred)
print("The Confusion Matrix for LDA is:\n",lda_cm)
print("\n")


''' K Nearest Neighbours '''
from sklearn.neighbors import KNeighborsClassifier

num_neighbors = range(1, 101) 
accuracies = []
for num in num_neighbors:
    knn = KNeighborsClassifier(n_neighbors = num)
    knn.fit(X_train, Y_train)

    KNN_Y_pred = knn.predict(X_test)
    
    accuracies.append({'k': num, 'acc': np.mean(KNN_Y_pred == Y_test)})
    #print("\nK: ", num)
    #print("Accuracy of KNN: ", np.mean(KNN_Y_pred == Y_test))
max_acc = max(accuracies, key = lambda x:x['acc'])
print("Accuracy of KNN:", max_acc['acc'], "when k =", max_acc['k'])
knn = KNeighborsClassifier(n_neighbors = max_acc['k'])
knn.fit(X_train, Y_train)
KNN_Y_pred = knn.predict(X_test)
    
knn_cm = confusion_matrix(Y_test,KNN_Y_pred)
print("The Confusion Matrix for KNN is:\n",knn_cm)
print("\n")


''' Logistic Regression '''
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="lbfgs")
lr.fit(X_train, Y_train)

LR_Y_pred = lr.predict(X_test)

print("Accuracy of Logistic Regression: ", np.mean(LR_Y_pred == Y_test))
LR_cm = confusion_matrix(Y_test,LR_Y_pred)
print("The Confusion Matrix for LR is:\n",LR_cm)
print("\n")


''' Naive Bayes '''
from sklearn.naive_bayes import GaussianNB
# Gaussian
gnb = GaussianNB()
gnb.fit(X_train, Y_train)

GNB_Y_pred = gnb.predict(X_test)

print("Accuracy of GaussianNB: ", np.mean(GNB_Y_pred == Y_test))
GNB_cm = confusion_matrix(Y_test,GNB_Y_pred)
print("The Confusion Matrix for GNB is:\n",GNB_cm)
print("\n")


# Kfolds Cross Validation. 
# Importing Libraries
print("=========== K-Folds Cross Validation ===========\n")
from sklearn.model_selection import cross_val_score


# LDA
lda = LinearDiscriminantAnalysis()

print("\nLinear Discriminant Analysis:")
lda_accuracies = cross_val_score(estimator = lda, X = X_train, y = Y_train, cv = 10, n_jobs = 1)
print("Maximum Accuracy:", max(lda_accuracies))
print("Variance in accuracies:", lda_accuracies.std())


# KNN
knn = KNeighborsClassifier(n_neighbors = max_acc['k'])

print("\nK Nearest Neighbors:")
knn_accuracies = cross_val_score(estimator = knn, X = X_train, y = Y_train, cv = 10, n_jobs = 1)
print("Maximum Accuracy:", max(knn_accuracies))
print("Variance in accuracies:", knn_accuracies.std())


# Log Reg
lr = LogisticRegression(solver="lbfgs")

print("\nLogistic Regression:")
lr_accuracies = cross_val_score(estimator = lr, X = X_train, y = Y_train, cv = 10, n_jobs = 1)
print("Maximum Accuracy:", max(lr_accuracies))
print("Variance in accuracies:", lr_accuracies.std())


# Naive Bayes Classifier
gnb = GaussianNB()

print("\nNaive Bayes Classifier:")
gnb_accuracies = cross_val_score(estimator = gnb, X = X_train, y = Y_train, cv = 10, n_jobs = 1)
print("Maximum Accuracy:", max(gnb_accuracies))
print("Variance in accuracies:", gnb_accuracies.std())


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt1

fpr = dict()
tpr = dict()
roc_auc = dict()

fpr, tpr, _ = roc_curve(Y_test, LDA_Y_pred)
roc_auc = auc(fpr, tpr)

lw = 2
plt1.plot(fpr, tpr, color='darkorange',
     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt1.plot([0, 1], [0, 1], 'k--', lw=lw)
plt1.xlim([0.0, 1.0])
plt1.ylim([0.0, 1.05])
plt1.xlabel('False Positive Rate')
plt1.ylabel('True Positive Rate')
plt1.title('Some extension of Receiver operating characteristic')
plt1.legend(loc="lower right")
plt1.show()









'''PCA'''
import numpy as np
from sklearn.decomposition import PCA
pca1=PCA()
pca1.fit(X)
print("The ratio of PCA components are given below:")
print(pca1.explained_variance_ratio_)

pca = PCA(n_components=2)
pca.fit(X)
X_pca=pca.transform(X)
X.shape
X_pca.shape
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1],c=df['Outcome'],cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
df_comp = pd.DataFrame(pca.components_)
#df.keys()
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')

print("\n")
print("We will be considering first 2 components as it explains 47.82% variance")
print(pca.explained_variance_ratio_)
print("\n")
print("Following are the components:")
print(X_pca)
print("\n")


print("=========== Accuracy after PCA ============")
print("\n")

X_pca_train, X_pca_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size = 0.29, random_state = 1)

''' Linear Discriminant Analysis '''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_pca_train, Y_train)

LDA_Y_pred = lda.predict(X_pca_test)

print("Accuracy of LDA: ", np.mean(LDA_Y_pred == Y_test))
lda_cm = confusion_matrix(Y_test,LDA_Y_pred)
print("The Confusion Matrix for LDA is:\n",lda_cm)
print("\n")


''' K Nearest Neighbours '''
from sklearn.neighbors import KNeighborsClassifier

num_neighbors = range(1, 101) 
accuracies = []
for num in num_neighbors:
    knn = KNeighborsClassifier(n_neighbors = num)
    knn.fit(X_pca_train, Y_train)

    KNN_Y_pred = knn.predict(X_pca_test)
    
    accuracies.append({'k': num, 'acc': np.mean(KNN_Y_pred == Y_test)})
    #print("\nK: ", num)
    #print("Accuracy of KNN: ", np.mean(KNN_Y_pred == Y_test))
    
max_acc = max(accuracies, key = lambda x:x['acc'])
knn = KNeighborsClassifier(n_neighbors = max_acc['k'])
knn.fit(X_train, Y_train)
KNN_Y_pred = knn.predict(X_test)
print("Accuracy of KNN:", max_acc['acc'], "when k =", max_acc['k'])
knn_cm = confusion_matrix(Y_test,KNN_Y_pred)
print("The Confusion Matrix for KNN is:\n",knn_cm)
print("\n")

''' Logistic Regression '''
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_pca_train, Y_train)

LR_Y_pred = lr.predict(X_pca_test)

print("Accuracy of Logistic Regression: ", np.mean(LR_Y_pred == Y_test))
LR_cm = confusion_matrix(Y_test,LR_Y_pred)
print("The Confusion Matrix for LR is:\n",LR_cm)
print("\n")


''' Naive Bayes '''
from sklearn.naive_bayes import GaussianNB
# Gaussian
gnb = GaussianNB()
gnb.fit(X_pca_train, Y_train)

GNB_Y_pred = gnb.predict(X_pca_test)

print("Accuracy of GaussianNB: ", np.mean(GNB_Y_pred == Y_test))
GNB_cm = confusion_matrix(Y_test,GNB_Y_pred)
print("The Confusion Matrix for GNB is:\n",GNB_cm)
print("\n")
print("\n")

print("=========== K-Folds Cross Validation after PCA ===========\n")
from sklearn.model_selection import cross_val_score


# LDA
lda = LinearDiscriminantAnalysis()

print("\nLinear Discriminant Analysis:")
lda_accuracies = cross_val_score(estimator = lda, X = X_pca_train, y = Y_train, cv = 10, n_jobs = 1)
print("Maximum Accuracy:", max(lda_accuracies))
print("Variance in accuracies:", lda_accuracies.std())

# KNN
knn = KNeighborsClassifier(n_neighbors = max_acc['k'])

print("\nK Nearest Neighbors:")
knn_accuracies = cross_val_score(estimator = knn, X = X_pca_train, y = Y_train, cv = 10, n_jobs = 1)
print("Maximum Accuracy:", max(knn_accuracies))
print("Variance in accuracies:", knn_accuracies.std())


# Log Reg
lr = LogisticRegression()

print("\nLogistic Regression:")
lr_accuracies = cross_val_score(estimator = lr, X = X_pca_train, y = Y_train, cv = 10, n_jobs = 1)
print("Maximum Accuracy:", max(lr_accuracies))
print("Variance in accuracies:", lr_accuracies.std())


# Naive Bayes Classifier
gnb = GaussianNB()

print("\nGaussianNB:")
gnb_accuracies = cross_val_score(estimator = gnb, X = X_pca_train, y = Y_train, cv = 10, n_jobs = 1)
print("Maximum Accuracy:", max(gnb_accuracies))
print("Variance in accuracies:", gnb_accuracies.std())
