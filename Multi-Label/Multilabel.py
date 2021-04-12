import scipy
from scipy.io import arff
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.svm import SVC
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import seaborn as sns

def plot_data(df):

    #LABEL COUNT
    flag = False
    if flag:
        counts = []
        labels = ['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised']
        for i in labels:
            print(i)
            c = df[i].value_counts()
            counts.append(c[1])
        print(c[1])
        print(counts)
        Number_of_data = pd.DataFrame(list(zip(labels, counts)), columns=['Label', 'Number of Data'])
        Number_of_data.plot(x="Label", y="Number of Data", kind="bar")
        plt.show()
    else:
        # SONGS HAVING MULTIPLE LABELS
        rowSums = df.loc[:,'amazed-suprised':'angry-aggresive'].sum(axis=1)
        multiLabel_counts = rowSums.value_counts()
        #print(multiLabel_counts.sort_index())
        #multiLabel_counts.sort_index().plot(x="number of labels",y="Number of Songs",kind="bar")
        #plt.show()

        multiLabel_counts = multiLabel_counts.sort_index()
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
        plt.title("Songs having multiple labels ")
        plt.ylabel('Number of songs', fontsize=18)
        plt.xlabel('Number of labels', fontsize=18)
        # adding the text labels
        rects = ax.patches
        labels = multiLabel_counts.values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
        plt.show()



def clean_data(data):

    df = pd.DataFrame(data)

    # print(df[['angry-aggresive','sad-lonely','quiet-still','relaxing-calm','happy-pleased','amazed-suprised']])
    df = df.replace(b'0', 0)
    df = df.replace(b'1', 1)

    plot_data(df)


    X = df.drop(
        columns=['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised'])
    #X = X[['BHSUM3','BHSUM2','BHSUM1','BH_HighLowRatio','BH_HighPeakBPM','BH_HighPeakAmp','BH_LowPeakBPM','BH_LowPeakAmp']]
    X = StandardScaler().fit_transform(X)

    y = df[['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised']]



    return X,y

def binary(X_train, X_test, y_train, y_test):

    print("Binary Relevance")
    model = BinaryRelevance(classifier=SVC(), require_dense=[True, True]).fit(X_train, y_train)
    print("Hamming: {}".format(hamming_loss(y_test, model.predict(X_test))))
    print("Accuracy: {}".format(accuracy_score(y_test, model.predict(X_test))))
    print("Precision: " + str(precision_score(y_test, model.predict(X_test), average="micro")))
    print("\n")

def chain(X_train, X_test, y_train, y_test):

    print("Classifier chain")
    model = ClassifierChain(LogisticRegression()).fit(X_train,y_train)
    print("Hamming: "+str(hamming_loss(y_test, model.predict(X_test))))
    print("Accuracy: "+str(accuracy_score(y_test, model.predict(X_test))))
    print("Precision: " + str(precision_score(y_test, model.predict(X_test), average="micro")))
    print("\n")

def powerset(X_train, X_test, y_train, y_test):

    print("Label Powerset")
    model = LabelPowerset(SVC()).fit(X_train,y_train)
    print("Hamming: " + str(hamming_loss(y_test, model.predict(X_test))))
    print("Accuracy: " + str(accuracy_score(y_test, model.predict(X_test))))
    print("Precision: " + str(precision_score(y_test, model.predict(X_test),average="weighted")))
    print("\n")

def MlKnn(X_train, X_test, y_train, y_test):

    X_train = lil_matrix(X_train).toarray()
    y_train = lil_matrix(y_train).toarray()
    X_test = lil_matrix(X_test).toarray()
    y_test = lil_matrix(y_test).toarray()

    print("MlKnn")
    model = MLkNN(k=5).fit(X_train,y_train)
    print("Hamming: " + str(hamming_loss(y_test, model.predict(X_test))))
    print("Accuracy: " + str(accuracy_score(y_test, model.predict(X_test))))

def oneRest(X_train, X_test, y_train, y_test):

    print("One vs Rest")
    pipe = Pipeline([('clf', OneVsRestClassifier(RandomForestClassifier(), n_jobs=-1))])
    categories = ['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised']



    for i in categories:
        k=0;
        pipe.fit(X_train, y_train[i])
        prediction = pipe.predict(X_test)
        print('Test accuracy is {}'.format(accuracy_score(y_test[i], prediction)))
        print('Hamming Loss is {}'.format(hamming_loss(y_test[i], prediction)))




def keras():
    print("Keras")

if __name__ == '__main__':
    data, meta = scipy.io.arff.loadarff(r'emotions.arff')
    X,y = clean_data(data)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
    #X_train = lil_matrix(X_train).toarray()
    #y_train = lil_matrix(y_train).toarray()
    #X_test = lil_matrix(X_test).toarray()
    #y_test = lil_matrix(y_test).toarray()

    #binary(X_train, X_test, y_train, y_test)
    #chain(X_train, X_test, y_train, y_test)
    #powerset(X_train, X_test, y_train, y_test)
    #MlKnn(X_train, X_test, y_train, y_test)
    #oneRest(X_train, X_test, y_train, y_test)









