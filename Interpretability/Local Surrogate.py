import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from graphviz import Source
from IPython.display import SVG
from IPython.display import display
from sklearn.metrics import accuracy_score
from ipywidgets import interactive
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing


def dataset(data):
    # Dropping null rows and unwanted columns
    data = data.dropna()
    # data = data.sample(20000)
    genres = ['Alternative', 'Jazz', 'Pop', 'Electronic', 'Hip-Hop', 'Rock']
    data = data.loc[data['genre'].isin(genres)]

    # selected = choose_instance(data)

    # Encoding
    data['mode'] = data['mode'].replace({'Major': 1, 'Minor': 0}).astype(int)

    # print(data[data["track_name"] == 'Creep'])

    scaled = ['duration_ms', 'popularity', "tempo", 'loudness']
    data[scaled] = MinMaxScaler().fit_transform(data[scaled])

    data = data.drop(columns=['artist_name', 'track_name', 'key', 'time_signature', 'track_id'])
    data['genre'] = data['genre'].apply(lambda x: x.replace(" ", "_"))

    return data


def decision_plot(new_X_train2, new_y_train2, feature_names, test, model):
    dt = DecisionTreeClassifier(random_state=0
                                , criterion='entropy'
                                , max_depth=1)

    dt.fit(new_X_train2, new_y_train2)
    print("Decision Tree Predicts for Instance:" + str(
        dt.predict(test)) + " and Random Forests predicted:" + str(model.predict(test)))
    fidelityPreds = dt.predict(new_X_train2)
    print("Let's see fidelity", accuracy_score(new_y_train2, fidelityPreds))

    graph = Source(export_graphviz(dt
                                   , out_file=None
                                   , feature_names=feature_names
                                   , class_names=dt.classes_
                                   , filled=True))
    display(SVG(graph.pipe(format='svg')))
    print("Lets find out the path for this specific instance!")
    for i in dt.decision_path(test):
        print(i)
    return dt




def Local_Interpretability(data):
    # for i in data['genre'].unique():
    #   new_data = data.copy(deep=True)
    #  new_data['genre'] = new_data['genre'].apply(lambda y: y if y == i else "Not_" + i)
    # X = new_data.drop(columns=['genre'])
    # y = new_data['genre']
    X = data.drop(columns=['genre'])
    y = data['genre']
    num_classes = y.unique()
    # y = np.ravel(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape)

    model = RandomForestClassifier().fit(X_train, y_train)
    new_y_train = model.predict(X_train)
    print(new_y_train)
    new_X_train = X_train

    classifier = KNeighborsClassifier(n_neighbors=50, weights="distance", metric="minkowski", p=2)
    classifier = classifier.fit(new_X_train, new_y_train)

    test = [X_test[2]]
    test2 = X_test[2]
    real = y_test.to_numpy()
    print("--The real Category of the instance is: " + str(real[2]))

    new_y = classifier.kneighbors(test, n_neighbors=200, return_distance=False)

    new_y_train2 = []
    new_X_train2 = []
    for i in new_y[0]:
        new_y_train2.append(new_y_train[i])
        new_X_train2.append(new_X_train[i])

    # Decision Tree
    dt = decision_plot(new_X_train2, new_y_train2, X.columns.values, test, model)
    inter = interactive(dt, depth=(1, 5))
    # display(inter)

    lr = LogisticRegression(solver="newton-cg", penalty='l2', max_iter=100, C=100, random_state=0).fit(new_X_train2,
                                                                                                       new_y_train2)

    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(new_X_train2), feature_names=X.columns.values,
                                                       class_names=dt.classes_)
    exp = explainer.explain_instance(test2, dt.predict_proba)
    exp.as_pyplot_figure()
    plt.show()
    #exp.show_in_notebook(show_table=True, show_all=False)



def onevsone(data):
    labels = data['genre'].unique()
    for i in labels:
        for j in labels:
            if i != j:
                first = data.loc[data['genre'] == i]
                second = data.loc[data['genre'] == j]
                new_data = pd.concat([first, second])
                Local_Interpretability(new_data)


if __name__ == '__main__':
    data = pd.read_csv(r'SpotifyFeatures.csv')

    data = dataset(data)
    onevsone(data)