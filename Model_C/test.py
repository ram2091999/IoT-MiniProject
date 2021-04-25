import sys
import pandas as pd
import numpy as np
import random
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from glob import iglob
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix


def load_mal_data():
    df_mirai = pd.concat((pd.read_csv(f) for f in iglob('../data/**/mirai_attacks/*.csv', recursive=True)), ignore_index=True)
    df_gafgyt = pd.DataFrame()
    for f in iglob('../data/**/gafgyt_attacks/*.csv', recursive=True):
        df_gafgyt = df_gafgyt.append(pd.read_csv(f), ignore_index=True)
    return df_mirai.append(df_gafgyt)


def test(top_n_features = 115):
    test_with_data(top_n_features, load_mal_data())

def test_with_data(top_n_features, df_malicious):

    # TESTING PHASE
    df = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher = pd.read_csv('../fisher.csv')
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]
    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17), [int(1/3*len(df)), int(2/3*len(df))])
    scaler = StandardScaler()
    scaler.fit(x_train.append(x_opt))
    
    # Loading the model
    saved_model = load_model(f'models/model_{top_n_features}.h5')
    with open(f'threshold_{top_n_features}') as t:
        tr = np.float64(t.read())
    print(f"Calculated threshold - {tr}")
    model = AnomalyModel(saved_model, tr, scaler)

    df_benign = pd.DataFrame(x_test, columns=df.columns)
    df_benign['malicious'] = 0
    df_malicious = df_malicious.sample(n=df_benign.shape[0], random_state=17)[list(features)]
    df_malicious['malicious'] = 1
    df = df_benign.append(df_malicious)
    X_test = df.drop(columns=['malicious']).values
    X_test_scaled = scaler.transform(X_test)
    Y_test = df['malicious']
    Y_pred = model.predict(X_test_scaled)

    # Accuracy of prediction
    print('Accuracy')
    print(accuracy_score(Y_test, Y_pred))

    # Recall score
    print('Recall')
    print(recall_score(Y_test, Y_pred))

    # Precision score
    print('Precision')
    print(precision_score(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

class AnomalyModel:
    def __init__(self, model, threshold, scaler):
        self.model = model
        self.threshold = threshold
        self.scaler = scaler

    def predict(self, x):
        x_pred = self.model.predict(x)
        mse = np.mean(np.power(x - x_pred, 2), axis=1)
        y_pred = mse > self.threshold
        return y_pred.astype(int)

    def scale_predict_classes(self, x):
        x = self.scaler.transform(x)
        y_pred = self.predict(x)
        classes_arr = []
        for e in y_pred:
            el = [0,0]
            el[e] = 1
            classes_arr.append(el)

        return np.array(classes_arr)


if __name__ == '__main__':
    test(*sys.argv[1:])