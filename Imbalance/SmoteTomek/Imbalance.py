import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTETomek
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans



def dataset(data):
    #Dropping null rows and unwanted columns
    data = data.dropna()
    data = data[data.genre != 'A Capella']

    #Converting Children's music to one unified category
    data.loc[(data.genre == 'Children\'s Music'), 'genre'] = 'Music for Children'
    data.loc[(data.genre == 'Children’s Music'), 'genre'] = 'Music for Children'

    data.loc[(data.genre == 'Reggaeton'), 'genre'] = 'Reggae'


    #Encoding
    data['mode'] = data['mode'].replace({'Major' : 1, 'Minor' : 0}).astype(int)

    #print(data[data["track_name"] == 'Creep'])

    scaled = ['duration_ms', 'popularity', "tempo", 'loudness']
    data[scaled] = MinMaxScaler().fit_transform(data[scaled])

    data = data.drop(columns=['artist_name', 'track_name','key', 'time_signature','track_id'])
    data['genre'] = data['genre'].apply(lambda x: x.replace(" ", "_"))

    return data

def reportToArray(report):
    dict = {}
    #print(report.split('\n'))
    ind = [2,3,5]
    for i in ind:
        temp = report.split('\n')[i]
        arr = []
        temp = temp.replace("avg / total","avg/total")
        temp = ' '.join(temp.split())

        key = temp.split(' ')[0]
        #print(key)
        for j in range(1,len(temp.split(' '))):
            arr.append(float(temp.split(' ')[j]))

        dict[key] = arr
    return dict

def onevsrest(data,flag):
    if flag:
      ind = 0
      for i in data['genre'].unique():

        new_data = data.copy(deep=True)
        new_data['genre'] = new_data['genre'].apply(lambda y: y if y == i else "Not_" + i)
        X = new_data.drop(columns=['genre'])
        y = new_data['genre']

        #Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        # Before balancing
        pipeline = make_pipeline(LogisticRegression(solver="lbfgs"))
        pipeline.fit(X_train, y_train)

        res_before = reportToArray(classification_report_imbalanced(y_test, pipeline.predict(X_test)))

        before = res_before['avg/total']
        allKeys = new_data["genre"].value_counts().index.tolist()
        allValues = new_data["genre"].value_counts().values



        # After balancing
        pipeline = make_pipeline(SMOTETomek(random_state=5),
                                 LogisticRegression(solver="lbfgs")).fit(X_train, y_train)

        res_after = reportToArray(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
        after = res_after['avg/total']

        #Results
        labels = ['Prec', 'Rec', 'Spec', 'F1', 'Geo', 'IBA', 'Sup (10^5)']

        after = res_after['avg/total']

        fig, axs = plt.subplots(1, 2, squeeze=False)
        axs[0, 0].pie(x=np.array(allValues).ravel(), labels=np.array(allKeys).ravel(), colors=['brown', 'gold'])
        axs[0, 0].set_title(
            str(allValues[1]) + " to " + str(allValues[0]) + "\n1 to " + str(round(allValues[0] / allValues[1], 2)))

        # axs[0, 1].pie(x=np.array(allValues).ravel(), labels=np.array(allKeys).ravel())
        # axs[0, 1].set_title('Axis [' + str(ind) + ', 0]')

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        print(len(x - width / 2))
        # print(type(np.array(before)))
        print(len(before))
        print(len(after))

        before[len(before) - 1] = round(before[len(before) - 1] / 100000, 2)
        after[len(after) - 1] = round(after[len(after) - 1] / 100000, 2)
        # before = [25, 32, 34, 20, 25]
        rects1 = axs[0, 1].bar(np.array(x - width / 2), np.array(before), width, label='Before', color="red")
        rects2 = axs[0, 1].bar(x + width / 2, after, width, label='After', color='green')

        axs[0, 1].set_ylabel('Scores')
        axs[0, 1].set_title('Metric Scores Before and After Balancing')
        axs[0, 1].set_xticks(x)
        axs[0, 1].set_xticklabels(labels)
        axs[0, 1].legend()
        axs[0, 1].bar_label(rects1, padding=3)
        axs[0, 1].bar_label(rects2, padding=3)

        # plt.show()
        fig.set_size_inches(14, 8)
        #plt.savefig(i + ".png")





def onevsone(data,flag):
    if flag:
        labels = data['genre'].unique()

        label = 'genre'
        allX = []
        allY = []
        for c in data[label].unique():
            category = data.loc[data[label] == c]
        allX.append(category.median().to_numpy())
        allY.append(c)
        kmeans = KMeans(n_clusters=5, random_state=0).fit(allX)
        allnewY = kmeans.labels_
        print(len(allY), len(allnewY))

        for i in range(len(allY)):
            print(allY[i], allnewY[i])
            data.loc[(data.genre == allY[i]), 'genre'] = allnewY[i]
        print(data['genre'].value_counts())

        for i in labels:
            for j in labels:
                if i!=j:

                    first = data.loc[data['genre'] == i]
                    second = data.loc[data['genre'] == j]
                    ndf = pd.concat([first,second])
                    y = ndf['genre']
                    X = ndf.drop(columns=['genre'])


                    # Train test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11,stratify=y)



                    # Pipeline
                    pipeline = make_pipeline(LogisticRegression(solver="lbfgs",max_iter=300)).fit(X_train, y_train)

                    res_before = reportToArray(classification_report_imbalanced(y_test, pipeline.predict(X_test)))

                    before = res_before['avg/total']
                    allKeys = ndf[label].value_counts().index.tolist()
                    allValues = ndf[label].value_counts().values
                    labels = ['Prec', 'Rec', 'Spec', 'F1', 'Geo', 'IBA', 'Sup (10^5)']

                    # Create a pipeline
                    pipeline = make_pipeline(SMOTETomek(random_state=3),
                                             LogisticRegression(solver="lbfgs",max_iter=300))
                    pipeline.fit(X_train, y_train)

                    # Classify and report the results
                    print("logistic regression with SMOTE")
                    res_after = reportToArray(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
                    # print(res_after)
                    after = res_after['avg/total']

                    fig, axs = plt.subplots(1, 2, squeeze=False)
                    axs[0, 0].pie(x=np.array(allValues).ravel(), labels=np.array(allKeys).ravel(),
                                  colors=['brown', 'gold'])
                    axs[0, 0].set_title(
                        str(allValues[1]) + " to " + str(allValues[0]) + "\n1 to " + str(
                            round(allValues[0] / allValues[1], 2)))

                    x = np.arange(len(labels))  # the label locations
                    width = 0.35  # the width of the bars

                    before[len(before) - 1] = round(before[len(before) - 1] / 100000, 2)
                    after[len(after) - 1] = round(after[len(after) - 1] / 100000, 2)
                    # before = [25, 32, 34, 20, 25]
                    rects1 = axs[0, 1].bar(np.array(x - width / 2), np.array(before), width, label='Before',
                                           color="red")
                    rects2 = axs[0, 1].bar(x + width / 2, after, width, label='After', color='green')

                    axs[0, 1].set_ylabel('Scores')
                    axs[0, 1].set_title('Metric Scores Before and After Balancing')
                    axs[0, 1].set_xticks(x)
                    axs[0, 1].set_xticklabels(labels)
                    axs[0, 1].legend()
                    axs[0, 1].bar_label(rects1, padding=3)
                    axs[0, 1].bar_label(rects2, padding=3)

                    # plt.show()
                    fig.set_size_inches(14, 8)
                    plt.savefig(str(i) + "vs" + str(j) + ".png")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data = pd.read_csv(r'SpotifyFeatures.csv')

    data = dataset(data)
    onevsrest(data,True)
    onevsone(data,False)




