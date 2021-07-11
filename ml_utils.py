from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import  KNeighborsClassifier
#from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# define a Gaussain NB classifier
clf_NB = GaussianNB()
#clf = KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=30, p=1, metric='minkowski')
#clf = LinearRegression()
clf_SVC = SVC(kernel='poly', degree=3, max_iter=300000)

accurate_clf= None#placeholder to contain the final value of the hightest accurate classifier

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    global accurate_clf
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf_NB.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc_NB = accuracy_score(y_test, clf_NB.predict(X_test))
    print(f"Navie Bayes Model trained with accuracy: {round(acc_NB, 3)}")

    clf_SVC.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc_SVC = accuracy_score(y_test, clf_SVC.predict(X_test))
    print(f"SVC Model trained with accuracy: {round(acc_SVC, 3)}")

    accurate_clf = clf_NB if (acc_NB>acc_SVC) else clf_SVC

    print(f"Selected classification Model:",type(accurate_clf))
    
# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = accurate_clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    accurate_clf = clf_NB
    accurate_clf.fit(X, y)
