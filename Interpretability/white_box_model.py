import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from pdpbox import pdp, get_dataset, info_plots
import random as rnd
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
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
def modelandplotTree(x_train, x_test, y_train, y_test):
    model = DecisionTreeClassifier(criterion='gini',max_depth=2, random_state=0)
    model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_test)
    feature_names = X.columns
    fig = plt.figure(dpi=200)
    _ = tree.plot_tree(model,
                       feature_names=X.columns,
                       class_names=[first_genre,second_genre],
                       filled=True)
    fig.savefig(file+first_genre+" "+second_genre+" - DT.png",transparent=True)

    weights = model.feature_importances_
    model_weights = pd.DataFrame({ 'features': list(feature_names),'weights': list(weights)})
    model_weights = model_weights.sort_values(by='weights', ascending=False)
    fig = plt.figure(figsize=(8, 6), dpi=200)
    sns.barplot(x="weights", y="features", data=model_weights,palette="dark:salmon_r").set_title(first_genre+" vs "+second_genre)
    plt.xticks(rotation=90)
    fig.savefig(file+first_genre+" "+second_genre+" - DT - weights.png",transparent=True)

    # for feature_txt in list(feature_names):
    #     df = pd.DataFrame(X, columns=list(feature_names))
    #     pdp_goals = pdp.pdp_isolate(model=model, dataset=df, model_features=df.columns, feature=feature_txt)
    #
    #
    #     # plot it
    #     #fig = plt.figure(figsize=(8, 6), dpi=200)
    #     pdp.pdp_plot(pdp_goals, feature_txt)
    #     #plt.show()
    #     pdp.plt.savefig(file+"weights/"+first_genre+" "+second_genre+" - pdp -- "+feature_txt+".png",transparent=True)


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
df = df.sample(20000,random_state=0)

genres_kept = ['Alternative', 'Jazz', 'Pop', 'Electronic','Hip-Hop','Rock']
df[label] = df[label].dropna()

file = "WHITE/"
for first_genre in genres_kept:
    for second_genre in genres_kept:
        if(first_genre != second_genre and second_genre[0] >= first_genre[0]):
            ndf = df.copy(deep=True)
            ndf[label] = ndf.loc[df[label].isin([first_genre,second_genre])]
            ndf = ndf.dropna()
            X = ndf.drop([label], axis=1)
            y = ndf[label]
            x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,
                                                                                random_state=101,stratify=y)
            X = ndf.drop([label], axis=1)
            y = ndf[label]
            x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,random_state=101,stratify=y)

            modelandplotTree(x_train, x_test, y_train, y_test)
