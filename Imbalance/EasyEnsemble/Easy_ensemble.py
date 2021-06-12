import pandas as pd

import imblearn
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,KMeansSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from collections import Counter
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score,f1_score
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn import tree,datasets,model_selection,metrics,ensemble,tree,linear_model,svm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
df = pd.read_csv("SpotifyFeatures.csv")

df = df.drop(['artist_name','track_name','track_id','key','time_signature'],axis=1)
df = df.dropna()
df['mode'] = df['mode'].apply(lambda x: 1 if x == 'Major' else 0)
df.loc[(df.genre == 'Children\'s Music'),'genre']= 'Music_for_Children'
df.loc[(df.genre == 'Childrenβ€™s Music'),'genre']= 'Music_for_Children'

df.loc[(df.genre == 'Reggaeton'),'genre']= 'Reggae'

df['genre'] = df['genre'].apply(lambda x: x.replace(" ","_"))
scaled = ['duration_ms','popularity',"tempo",'loudness']
df[scaled] = MinMaxScaler().fit_transform(df[scaled])
label = 'genre'
def reportToArray(report):
    dict = {}
    ind = [2,3,5]
    for i in ind:
        temp = report.split('\n')[i]
        arr = []
        temp = temp.replace("avg / total","avg/total")
        temp = ' '.join(temp.split())
        key = temp.split(' ')[0]
        for j in range(1,len(temp.split(' '))):
            arr.append(float(temp.split(' ')[j]))

        dict[key] = arr
    return dict
from imblearn.pipeline import make_pipeline

from imblearn.metrics import classification_report_imbalanced

print(df[label].unique())
#exit()
# #ONE VS REST
# ind = 0
# #for c in ['Electronic', 'Hip-Hop']: #df[label].unique():
# for c in df[label].unique():
#     #continue
#
#     ndf = df.copy(deep=True)
#     ndf[label] = ndf[label].apply(lambda x: x if x == c else "NOT_" + c)
#
#     #print(ndf[label].value_counts())
#     X = ndf.drop([label], axis=1)
#     y = ndf[label]
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,
#                                                                         random_state=101,stratify=y)
#
#     #model = ensemble.RandomForestClassifier(max_depth=1)
#     pipeline = make_pipeline(LogisticRegression(solver="lbfgs"))
#     pipeline.fit(x_train, y_train)
#
#     # Classify and report the results
#     print("logistic regression")
#     #res = classification_report_imbalanced(y_test, pipeline.predict(x_test))
#     #print(classification_report_imbalanced(y_test, pipeline.predict(x_test)))
#
#     #print(reportToArray(classification_report_imbalanced(y_test, pipeline.predict(x_test))))
#     res_before = reportToArray(classification_report_imbalanced(y_test, pipeline.predict(x_test)))
#
#     before = res_before['avg/total']
#     allKeys = ndf[label].value_counts().index.tolist()
#     allValues= ndf[label].value_counts().values
#
#
#
#
#
#
#     labels = ['Prec','Rec','Spec','F1','Geo','IBA','Sup (10^5)']
#     #print(res.split('avg / total')[1])
#
#     # Create a pipeline
#     # pipeline = make_pipeline(LogisticRegression(solver="lbfgs", class_weight='balanced'))
#     # pipeline.fit(x_train, y_train)
#     #
#     # # Classify and report the results
#     # print("logistic regression with balanced weights")
#     # print(classification_report_imbalanced(y_test, pipeline.predict(x_test)))
#
#     #Create a pipeline
#     #pipeline = make_pipeline(BorderlineSMOTE(random_state=3, k_neighbors=5),
#     #                         LogisticRegression(solver="lbfgs"))
#     pipeline = make_pipeline(EasyEnsembleClassifier(n_estimators=10, base_estimator=LogisticRegression(solver="lbfgs")))
#     #pipeline = make_pipeline(RUSBoostClassifier(n_estimators=10, base_estimator=LogisticRegression(solver="lbfgs")))
#     #pipelline = BalancedRandomForestClassifier()
#     pipeline.fit(x_train, y_train)
#
#     # Classify and report the results
#     print("logistic regression with EasyEnsemble")
#     res_after = reportToArray(classification_report_imbalanced(y_test, pipeline.predict(x_test)))
#     #print(res_after)
#     after = res_after['avg/total']
#
#     fig, axs = plt.subplots(1, 2, squeeze=False)
#     axs[0, 0].pie(x=np.array(allValues).ravel(), labels=np.array(allKeys).ravel(),colors=['brown','gold'])
#     axs[0, 0].set_title(
#         str(allValues[1]) + " to " + str(allValues[0]) + "\n1 to " + str(round(allValues[0] / allValues[1], 2)))
#
#     # axs[0, 1].pie(x=np.array(allValues).ravel(), labels=np.array(allKeys).ravel())
#     # axs[0, 1].set_title('Axis [' + str(ind) + ', 0]')
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.35  # the width of the bars
#
#     print(len(x - width / 2))
#     #print(type(np.array(before)))
#     print(len(before))
#     print(len(after))
#
#     before[len(before)-1] = round(before[len(before)-1]/100000,2)
#     after[len(after) - 1] = round(after[len(after) - 1] /100000,2)
#     #before = [25, 32, 34, 20, 25]
#     rects1 = axs[0, 1].bar(np.array(x - width / 2), np.array(before), width, label='Before',color="red")
#     rects2 = axs[0, 1].bar(x + width / 2, after, width, label='After',color='green')
#
#     axs[0, 1].set_ylabel('Scores')
#     axs[0, 1].set_title('Metric Scores Before and After Balancing')
#     axs[0, 1].set_xticks(x)
#     axs[0, 1].set_xticklabels(labels)
#     axs[0, 1].legend()
#     axs[0, 1].bar_label(rects1, padding=3)
#     axs[0, 1].bar_label(rects2, padding=3)
#
#     #plt.show()
#     fig.set_size_inches(14, 8)
#     plt.savefig("EASYENSEMBLE/OVReesnew_"+c+".png")


'''
Clustering to change the number of groups by 25 to 5
'''
allX = []
allY = []
for c in df[label].unique():
    category = df.loc[df[label] == c]
    allX.append(category.median().to_numpy())
    allY.append(c)


kmeans = KMeans(n_clusters=5, random_state=0).fit(allX)
allnewY = kmeans.labels_
print(len(allY),len(allnewY))

for i in range(len(allY)):
    print(allY[i],allnewY[i])
    df.loc[(df.genre == allY[i]),'genre']= allnewY[i]

print(df['genre'].value_counts())


print("one vs one")

#ONE VS ONE
for c in df[label].unique():
    for c2 in df[label].unique():
        if(c != c2 ):
            print(c,"VS",c2)
            first = df.loc[df[label] == c]
            second = df.loc[df[label] == c2]

            ndf = first.append(second,ignore_index=True)
            print(ndf[label].value_counts())
            X = ndf.drop([label], axis=1)
            y = ndf[label]
            x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,
                                                                                    random_state=101,stratify=y)

            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            pipeline = make_pipeline(LogisticRegression(solver="lbfgs"))
            pipeline.fit(x_train, y_train)
            print("logistic regression")
            res_before = reportToArray(classification_report_imbalanced(y_test, pipeline.predict(x_test)))

            before = res_before['avg/total']
            allKeys = ndf[label].value_counts().index.tolist()
            allValues= ndf[label].value_counts().values
            labels = ['Prec','Rec','Spec','F1','Geo','IBA','Sup (10^5)']


            pipeline = make_pipeline(EasyEnsembleClassifier(n_estimators=10, base_estimator=LogisticRegression(solver="lbfgs")))
            pipeline.fit(x_train, y_train)

            # Classify and report the results
            print("logistic regression with EasyEnsemble")
            res_after = reportToArray(classification_report_imbalanced(y_test, pipeline.predict(x_test)))
            after = res_after['avg/total']

            fig, axs = plt.subplots(1, 2, squeeze=False)
            axs[0, 0].pie(x=np.array(allValues).ravel(), labels=np.array(allKeys).ravel(),colors=['brown','gold'])
            axs[0, 0].set_title(
                str(allValues[1]) + " to " + str(allValues[0]) + "\n1 to " + str(round(allValues[0] / allValues[1], 2)))

            x = np.arange(len(labels))
            width = 0.35


            before[len(before)-1] = round(before[len(before)-1]/100000,2)
            after[len(after) - 1] = round(after[len(after) - 1] /100000,2)
            rects1 = axs[0, 1].bar(np.array(x - width / 2), np.array(before), width, label='Before',color="red")
            rects2 = axs[0, 1].bar(x + width / 2, after, width, label='After',color='green')

            axs[0, 1].set_ylabel('Scores')
            axs[0, 1].set_title('Metric Scores Before and After Balancing')
            axs[0, 1].set_xticks(x)
            axs[0, 1].set_xticklabels(labels)
            axs[0, 1].legend()
            axs[0, 1].bar_label(rects1, padding=3)
            axs[0, 1].bar_label(rects2, padding=3)

            fig.set_size_inches(14, 8)
            plt.savefig("EASYENSEMBLE/OVOeesnew_"+str(c)+"vs"+str(c2)+".png")
