import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn import svm
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
#from sklearn.linear_model import LinearRegression

df = pd.read_csv('t1.csv')
df.head()

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


y = df.label

X = vectorizer.fit_transform(df.tweet)

print(y.shape)
print(X.shape)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

#Naive bayes
lm = naive_bayes.MultinomialNB()
lm.fit(X_train,y_train)

a=roc_auc_score(y_test, lm.predict(X_test))

array = (["bahubali  was a disgusting movie"])
vector = vectorizer.transform(array)
print (lm.predict(vector))
print("\n \nNaive bayes:")
print(a)


sns.pairplot(df)

#Linear regression
cd = LinearRegression()
cd.fit(X_train,y_train)
a=roc_auc_score(y_test, cd.predict(X_test))
print("\n \nLinear Regression:")
print(a)



#logistic regression
model_logit = LogisticRegression()
model_logit.fit(X_train, y_train)
a=roc_auc_score(y_test, model_logit.predict(X_test))
print("\n \nlogistic regression:")
print(a)


# Decision Trees
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
a=roc_auc_score(y_test, classifier.predict(X_test))
print("\n \nDecision tree:")
print(a)


#support vector machine
cla = svm.SVC()
cla.fit(X_train, y_train)
#svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
a=roc_auc_score(y_test, cla.predict(X_test))
print("\n \nSVMS:")
print(a)




#support vector machine

print("\nplot cofusion maatrix for \n1.naive bayes \n2.linear regression\n3.logistic regression\n4.Decision Tree\n5.SVM\n")
x = int(input('enter the value:'))
if (x==1):
    print("\nNAIVE BAYES\n")
    predictions = lm.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
if (x==2):
    print("\nLINEAR REGRESSION\n")
    predictions = cd.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
if (x==3):
    print("\nLOGISTIC REGRESSION\n")
    predictions = model_logit.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
if (x==4):
    print("\nDECISION TREE\n")
    predictions = classifier.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
if(x==5):
    print("\nSUPPORT VECTOR MACHINE\n")
    predictions = cla.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

ch=1
while(ch==1):
    print("\nenter the sentence to check:\n");
    c = input()
    array = ([c])
    vector = vectorizer.transform(array)
    print("\nenter the training algorithm to be choosen \n1.naive bayes \n2.linear regression\n3.logistic regression\n4.Decision Tree\n5.SVM\n")
    x = int(input('enter the value:'))

    if (x==1):
        print("\nNAIVE BAYES\n")
        a=lm.predict(vector)
        print (a)
    if (x==2):
        print("\nLINEAR REGRESSION\n")
        a=cd.predict(vector)
        print (a)
    if (x==3):
        print("\nLOGISTIC REGRESSION\n")
        a=model_logit.predict(vector)
        print (a)
    if (x==4):
        print("\nDECISION TREE\n")
        a=classifier.predict(vector)
        print (a)
    if(x==5):
        print("\nSUPPORT VECTOR MACHINE\n")
        a=cla.predict(vector)
        print (a)

    if (a[0]==0):
        print("\nPOSITIVE!!!")
    else:
        print("\nNEGATIVE!!!")
    ch = int(input('Do you want to continue press 1 to continue'))
    
