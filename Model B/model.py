#imports
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


def main():

    # Train test split
    x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
        train_data_st, labels, test_size=0.25, random_state=42)

    # Create and fit model
    model2 = Sequential()
    model2.add(Dense(32, input_dim=train_data_st.shape[1], activation='relu'))
    model2.add(Dense(72, input_dim=train_data_st.shape[1], activation='relu'))
    model2.add(Dense(32, input_dim=train_data_st.shape[1], activation='relu'))
    model2.add(Dense(1, kernel_initializer='normal'))
    model2.add(Dense(labels.shape[1],activation='softmax'))
    model2.compile(loss='categorical_crossentropy', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                            patience=5, verbose=1, mode='auto')
    model2.fit(x_train_st,y_train_st,validation_data=(x_test_st,y_test_st),
            callbacks=[monitor],verbose=2,epochs=100)

    # Running eval          
    pred_st = model2.predict(x_test_st)
    pred_st_2 = np.argmax(pred_st,axis=1)
    y_eval_st_2 = np.argmax(y_test_st,axis=1)
    score_st_2 = metrics.accuracy_score(y_eval_st_2, pred_st_2)
    print("accuracy: {}".format(score_st_2))

    #Model 2 (Example)
    print(classification_report(y_eval_st_2, pred_st_2))

    
if __name__ == "__main__":
    main()   