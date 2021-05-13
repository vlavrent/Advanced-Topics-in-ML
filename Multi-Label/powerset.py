import scipy
from scipy.io import arff
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,coverage_error,label_ranking_average_precision_score,label_ranking_loss
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
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

def clean_data(data):

    df = pd.DataFrame(data)
    df = df.replace(b'0', 0)
    df = df.replace(b'1', 1)
    X = df.drop(
        columns=['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised'])
    X = StandardScaler().fit_transform(X)
    y = df[['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised']].to_numpy()

    return X,y

def binary(X_train, X_test, y_train, y_test):

    print("Binary Relevance")
    model = BinaryRelevance(classifier=SVC(), require_dense=[True, True]).fit(X_train, y_train)
    print("Hamming: {}".format(hamming_loss(y_test, model.predict(X_test))))
    print("Accuracy: {}".format(accuracy_score(y_test, model.predict(X_test))))
    print("\n")

def powerset(X_train, X_test, y_train, y_test):

    print("Label Powerset")
    model = LabelPowerset(SVC()).fit(X_train,y_train)
    y_pred = model.predict(X_test)

    hamming = hamming_loss(y_test, y_pred)
    subset_accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred,average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    coverage = coverage_error(y_test, y_pred.toarray())
    aps = label_ranking_average_precision_score(y_test, y_pred.toarray())
    rankingloss = label_ranking_loss(y_test, y_pred.toarray())
    print("Hamming: " + str(hamming))
    print("Subset Accuracy: " + str(subset_accuracy))
    print("Recall: " + str(recall))
    print("Precision: " + str(precision))
    print("F1: " + str(f1))
    print("Coverage error: "+str(coverage))
    print("Average Precision Score: "+str(aps))
    print("Ranking Loss: " + str(rankingloss))
    print("\n")

    return hamming,subset_accuracy,recall,precision,f1,coverage,aps,rankingloss


data, meta = scipy.io.arff.loadarff(r'emotions.arff')

df = pd.DataFrame(data)
df = df.replace(b'0', 0)
df = df.replace(b'1', 1)
#print(df)
labels = ['angry-aggresive', 'sad-lonely', 'quiet-still', 'relaxing-calm', 'happy-pleased', 'amazed-suprised']
# import numpy as np
# # def histogram_intersection(a, b):
# #     v = np.minimum(a, b).sum().round(decimals=1)
# #     return v
# ndf = df[labels]
# correlation_matrix = ndf.corr()#method=histogram_intersection)
# print(correlation_matrix)
# fig, ax = plt.subplots(3,2)
# count1 = count2 = 0
# fig.suptitle('Correlation of each category with each other', fontsize=16)
# for l in labels:
#     numberofdata = pd.DataFrame(list(zip(labels, correlation_matrix[l])), columns=['Label', l])
#     print((count1, count2))
#
#     empty =[k[0:6] for k in numberofdata['Label']]
#     if(count2==0):
#         ax[count1,count2].barh(numberofdata['Label'],numberofdata[l],color='brown')
#     else:
#         ax[count1, count2].barh(numberofdata['Label'], numberofdata[l], color='brown')
#     ax[count1,count2].set_title(l)
#
#     #ax[count1, count2].set_xlabel('Correlation')
#     count2+=1
#     if(count2 == 2):
#         count2=0
#         count1+=1
#     print(count1,count2)
# fig.set_size_inches(10, 10)
# plt.tight_layout()
# plt.savefig("corre.png")

X,y = clean_data(data)
kf = KFold()
hamming_arr = []
subset_accuracy_arr =[]
recall_arr = []
precision_arr = []
f1_arr = []
coverage_arr =[]
aps_arr = []
rankingloss_arr = []



#all_arr = [hamming_arr ,subset_accuracy_arr ,recall_arr ,precision_arr ,f1_arr ,coverage_arr ,aps_arr,rankingloss_arr]
all_arr = [hamming_arr ,subset_accuracy_arr ,recall_arr ,precision_arr ,f1_arr ]
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    hamming, subset_accuracy, recall, precision, f1, coverage, aps, rankingloss = powerset(X_train, X_test, y_train, y_test)
    hamming_arr.append(hamming)
    subset_accuracy_arr.append(subset_accuracy)
    recall_arr.append(recall)
    precision_arr.append(precision)
    f1_arr.append(f1)
    coverage_arr.append(coverage)
    aps_arr.append(aps)
    rankingloss_arr.append(rankingloss)

finalscores = []
for lst in all_arr:
    finalscores.append(np.array(lst).mean())

#objects = ('Hamming Loss', 'Subset Accuracy', 'Recall', 'Precision', 'f1', 'Coverage','Average Precision','Ranking Loss')
objects = ('Hamming Loss', 'Subset Accuracy', 'Recall', 'Precision', 'f1')
y_pos = np.arange(len(objects))

width = 0.35
fig, ax = plt.subplots()
b = ax.bar(y_pos, finalscores, color='brown',align='center')
ax.set_xticks(y_pos)
ax.set_xticklabels(objects)
ax.set_ylabel('Score')
ax.bar_label(b, fmt='%.3f')
plt.title('Metrics')
plt.show()






