import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import imblearn
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,KMeansSMOTE
from collections import Counter
from sklearn.svm import LinearSVC
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score,f1_score,accuracy_score
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn import tree,datasets,model_selection,metrics,ensemble,tree,linear_model,svm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
df = pd.read_csv("SpotifyFeatures.csv")

df = df.drop(['artist_name','track_name','track_id','key','time_signature'],axis=1)
df = df.dropna()
df['mode'] = df['mode'].apply(lambda x: 1 if x == 'Major' else 0)
df.loc[(df.genre == 'Children\'s Music'),'genre']= 'Music_for_Children'
df.loc[(df.genre == 'Children’s Music'),'genre']= 'Music_for_Children'

df.loc[(df.genre == 'Reggaeton'),'genre']= 'Reggae'

df['genre'] = df['genre'].apply(lambda x: x.replace(" ","_"))
#print(df['genre'].value_counts())
scaled = ['duration_ms','popularity',"tempo",'loudness']
df[scaled] = MinMaxScaler().fit_transform(df[scaled])
label = 'genre'
df = df.sample(20000,random_state=0)
#print(df)

#print(df['genre'].unique())

genres_kept = ['Alternative', 'Jazz', 'Pop', 'Electronic','Hip-Hop','Rock']
df[label] = df[label].dropna()

dictionaryGenres = {}
for g in genres_kept:
    dictionaryGenres[g] = 0


titles = []

allClassifiers = ["XGBoost","RandomForest","SVM"]

def chooseClassifier(classifier):
    if(classifier == "XGBoost"):
        model = XGBClassifier(random_state = 0)
    elif(classifier == "RandomForest"):
        model = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
    elif(classifier == "SVM"):
        model = LinearSVC(random_state = 0)
    return model



fidelAcc = {"XGBoost": [],"RandomForest":[],"SVM":[]}
fidelAccBef = {"XGBoost": [],"RandomForest":[],"SVM":[]}

for first_genre in genres_kept:
    for second_genre in genres_kept:
        allAcc = []
        allF1 = []
        if(first_genre != second_genre and second_genre[0] >= first_genre[0]):
            ndf = df.copy(deep=True)
            #ndf[label] = ndf[label].apply(lambda x: x if x == c else "NOT_" + c)
            ndf[label] = ndf.loc[df[label].isin([first_genre,second_genre])]
            ndf = ndf.dropna()

            for k in range(len(ndf[label].value_counts().index.values)):
                genre = ndf[label].value_counts().index.values[k]
                freq = ndf[label].value_counts()[k]
                dictionaryGenres[genre] = freq
            #print(ndf[label].value_counts())
            X = ndf.drop([label], axis=1)
            y = ndf[label]
            x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,
                                                                                random_state=101,stratify=y)
            X = ndf.drop([label], axis=1)
            y = ndf[label]
            x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,random_state=101,stratify=y)


            #train a black-box model
            print("adter predicting")
            for c in allClassifiers:
                tempAcc = []
                tempF1 = []
                classifier = chooseClassifier(c)
                classifier.fit(x_train, y_train)

                print("before predicting")
                y_pred = classifier.predict(x_test)

                new_x_train = x_train
                new_y_train = classifier.predict(x_train)

                #lin_model = LogisticRegression(solver="newton-cg",penalty='l2',max_iter=1000,C=100,random_state=0)
                lin_model = LogisticRegression(random_state=0)
                #lin_model = LogisticRegression(penalty='l1',max_iter=1000,C=100,random_state=0)
                lin_model.fit(new_x_train, new_y_train)
                #print("Simple Linear Model Performance:")
                fidelity  = accuracy_score(y_pred,lin_model.predict(x_test))
                #print("Fidelity",fidelity)
                #print("Accuracy in new data")
                accuracy_new_data = accuracy_score(y_test,lin_model.predict(x_test))
                fidelityf1  = f1_score(y_pred,lin_model.predict(x_test),average='micro')
                f1_new_data = f1_score(y_test,lin_model.predict(x_test),average='micro')

                feature_names = X.columns
                weights = lin_model.coef_
                model_weights = pd.DataFrame({ 'features': list(feature_names),'weights': list(weights[0])})
                model_weights = model_weights.reindex(model_weights['weights'].abs().sort_values(ascending=False).index) #Sort by absolute value
                model_weights = model_weights[(model_weights["weights"] != 0)]
                #print("Number of features:",len(model_weights.values))
                plt.figure(num=None, figsize=(12, 6), dpi=100,edgecolor='black')#, edgecolor='k')
                sns.barplot(x="weights", y="features", data=model_weights,palette="dark:salmon_r")
                #plt.title("Intercept (Bias): "+str(lin_model.intercept_[0])+"    "+first_genre+" vs "+second_genre,loc='center')
                plt.title(first_genre+" vs "+second_genre+" -- "+c,loc='center')
                plt.xticks(rotation=90)
                plt.savefig("INTER/"+first_genre+" vs "+second_genre+" "+c+".png", transparent=True)
                fig = plt.figure(num=None, figsize=(8, 6), dpi=100,edgecolor='black')#, edgecolor='k')
                sns.barplot(x=['accuracy','fidelity acc.','f1', 'fidelity f1'],y=[accuracy_new_data,fidelity,f1_new_data,fidelityf1],palette="dark:salmon_r").set_title(first_genre+" vs "+second_genre+" -- "+c)
                titles.append(first_genre+" vs "+second_genre)
                plt.savefig("INTER/BAR/"+first_genre+" vs "+second_genre+" "+c+".png", transparent=True)
                fidelity = round(fidelity,3)
                accuracy_new_data = round(accuracy_new_data,3)
                fidelAcc[c].append(fidelity)
                fidelAccBef[c].append(accuracy_new_data)



finalValsAcc = {"XGBoost": 0.0,"RandomForest":0.0,"SVM":0.0}
finalValsAccBef = {"XGBoost": 0.0,"RandomForest":0.0,"SVM":0.0}
for key, value in fidelAcc.items():
    m = np.mean(np.array(value))
    finalValsAcc[key] = m

for key, value in fidelAccBef.items():
    m = np.mean(np.array(value))
    finalValsAccBef[key] = m

print(finalValsAcc)
print(finalValsAccBef)





