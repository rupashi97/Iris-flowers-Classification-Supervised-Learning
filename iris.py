# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix 
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#Load dataset
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width','petal-length','petal-width','class']
dataset=read_csv(filename,names=names)

#shape
print(dataset.shape)

#head  -first 20 rows of data (Taking a peak)
print(dataset.head(10))

#descriptions - mean,max,min,std etc values of every attribute
print(dataset.describe())


#class distribution - no. of rows belonging to each class [we have 3 classes]
print(dataset.groupby('class').size())


"""
#Data Visualisation

# box and whisker plots (Univariate Plots)
dataset.plot(kind= ' box ' , subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms of classes
dataset.hist()
pyplot.show()

#scatter plot matrix - Multivariate plot
scatter_matrix(dataset)
pyplot.show()

"""

#Evaluating some algorithms

# Split-out validation dataset (80%-20%)
array = dataset.values
X = array[:,0:4]      #all rows of columns 0,1,2,3 -attributes
Y = array[:,4]        #all rows of column 4 -class
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)

# Spot-Check Algorithms
models = []
models.append(( ' LR ' , LogisticRegression()))
models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
models.append(( ' KNN ' , KNeighborsClassifier()))
models.append(( ' CART ' , DecisionTreeClassifier()))
models.append(( ' NB ' , GaussianNB()))
models.append(( ' SVM ' , SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=10, random_state=seed)    #dividing the training set into 10 folds/parts
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring= 'accuracy')
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)


 # Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Make predictions on validation dataset
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))          #93.3% total accuracy
print(confusion_matrix(Y_validation, predictions))        #indication of 3 errors made
print(classification_report(Y_validation, predictions))