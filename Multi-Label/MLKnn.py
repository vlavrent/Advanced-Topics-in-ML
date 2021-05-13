import scipy
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy.sparse import csr_matrix, lil_matrix
from skmultilearn.adapt import MLkNN
import seaborn as sns
import xgboost as xgb
import warnings
import numpy as np
from matplotlib import cm
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


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


        #my_colors = [(0.5, 0.4, 0.5), (0.75, 0.75, 0.25)] * 5  # <-- make two custom RGBs and repeat/alternate them over all the bar elements.
        sns.set(style="whitegrid", color_codes=True)
        color = cm.inferno_r(np.linspace(.3, .90, 10))


        Number_of_data = pd.DataFrame(list(zip(labels, counts)), columns=['Label', 'Number of Data'])

        Number_of_data.plot(x="Label", y="Number of Data", kind="bar",color=color,legend=None,figsize=(8,8))
        plt.gcf().subplots_adjust(bottom=0.02,top=0.5)
        plt.tight_layout()
        #plt.title("Label Count")
        plt.ylabel('Number of songs', fontsize=12)
        plt.xlabel('Labels', fontsize=12)
        plt.xticks(rotation=45)
        plt.savefig("LabelCounts.png",dpi=300, bbox_inches = "tight")
        plt.show()

    else:
        # SONGS HAVING MULTIPLE LABELS
        rowSums = df.loc[:,'amazed-suprised':'angry-aggresive'].sum(axis=1)
        multiLabel_counts = rowSums.value_counts()
        print(multiLabel_counts.sort_index())

        color = cm.inferno_r(np.linspace(.3, .90, 10))

        sns.set(style="whitegrid", color_codes=True)
        multiLabel_counts.sort_index().plot(x="number of labels",y="Number of Songs",kind="bar",color=color)
        #plt.title("Songs having multiple labels ")
        plt.ylabel('Number of songs', fontsize=12)
        plt.xlabel('Number of labels', fontsize=12)
        plt.savefig("NumberOfLabels.png", dpi=300, bbox_inches="tight")
        plt.show()
        print(len(df))



def clean_data(data):

    df = pd.DataFrame(data)

    # print(df[['angry-aggresive','sad-lonely','quiet-still','relaxing-calm','happy-pleased','amazed-suprised']])
    df = df.replace(b'0', 0)
    df = df.replace(b'1', 1)


    #plot_data(df)

    X = df.drop(
        columns=['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised'])
    #X = X[['BHSUM3','BHSUM2','BHSUM1','BH_HighLowRatio','BH_HighPeakBPM','BH_HighPeakAmp','BH_LowPeakBPM','BH_LowPeakAmp']]

    X = StandardScaler().fit_transform(X)


    y = df[['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised']].to_numpy()


    return X,y

def MLKnn_GridSearch(X_train,X_test,y_train,y_test):
    parameters = {'k': range(1, 5), 's': [ 0.2,0.3,0.4, 0.5,0.6, 0.7, 1.0]}
    score = 'f1_macro'

    clf = GridSearchCV(MLkNN(), parameters, scoring=score)
    clf.fit(X, y)

    print(clf.best_params_, clf.best_score_)


def MlKnn(X_train, X_test, y_train, y_test):

    X_train = lil_matrix(X_train).toarray()
    y_train = lil_matrix(y_train).toarray()
    X_test = lil_matrix(X_test).toarray()
    y_test = lil_matrix(y_test).toarray()



    print("MlKnn")
    model = MLkNN(k=3,s=0.2).fit(X_train,y_train)
    hamming = hamming_loss(y_test, model.predict(X_test))
    Subset_Accuracy = accuracy_score(y_test, model.predict(X_test))
    Precision = precision_score(y_test, model.predict(X_test), average="micro")
    Recall = recall_score(y_test,model.predict(X_test),average='micro')
    f1 = f1_score(y_test,model.predict(X_test),average='micro')

    print("Hamming: " + str(hamming_loss(y_test, model.predict(X_test))))
    print("Subset Accuracy: " + str(accuracy_score(y_test, model.predict(X_test))))
    print("Precision: " + str(precision_score(y_test, model.predict(X_test), average="micro")))
    print("Recall: " + str(recall_score(y_test,model.predict(X_test),average='micro')))
    print("F1 score: " + str(f1_score(y_test,model.predict(X_test),average='micro')))

    print("\n")


    return hamming,Subset_Accuracy,Precision,Recall, f1



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data, meta = scipy.io.arff.loadarff(r'emotions.arff')
    X,y = clean_data(data)


    kf = KFold(10)
    hamming_arr = []
    accuracy_arr = []
    recall_arr = []
    precision_arr = []
    f1_arr = []
    coverage_arr = []
    aps_arr = []
    rankingloss_arr = []
    grid = False
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if(grid):
            MLKnn_GridSearch(X_train,X_test,y_train,y_test)
        else:
            hamming, sub_accuracy, recall, precision, f1= MlKnn(X_train, X_test, y_train, y_test)
            hamming_arr.append(hamming)
            accuracy_arr.append(sub_accuracy)
            recall_arr.append(recall)
            precision_arr.append(precision)
            f1_arr.append(f1)
            #coverage_arr.append(coverage)
            #aps_arr.append(aps)
            #rankingloss_arr.append(rankingloss)

            finalscores = []
            all = [hamming_arr,accuracy_arr ,recall_arr ,precision_arr ,f1_arr ]
            for lst in all:
                finalscores.append(np.array(lst).mean())

    objects = ('Hamming Loss', 'Subset Accuracy', 'Recall', 'Precision', 'f1')
    y_pos = np.arange(len(objects))

    width = 0.35
    fig, ax = plt.subplots()
    b = ax.bar(y_pos, finalscores, color='brown', align='center')
    ax.set_xticks(y_pos)
    ax.set_xticklabels(objects)
    ax.set_ylabel('Score')
    ax.bar_label(b, fmt='%.3f')
    plt.title('Metrics')
    plt.show()

    #oneRest(X_train, X_test, y_train, y_test)
    #Xboost(X_train, X_test, y_train, y_test)









