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
    model = Sequential()
    model.add(Dense(10, input_dim=train_data_st.shape[1], activation='relu'))
    model.add(Dense(40, input_dim=train_data_st.shape[1], activation='relu'))
    model.add(Dense(10, input_dim=train_data_st.shape[1], activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(labels.shape[1],activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                            patience=5, verbose=1, mode='auto')
    model.fit(x_train_st,y_train_st,validation_data=(x_test_st,y_test_st),
            callbacks=[monitor],verbose=2,epochs=500)

    # Running eval          
    pred_st = model.predict(x_test_st)
    pred_st = np.argmax(pred_st,axis=1)
    y_eval_st = np.argmax(y_test_st,axis=1)
    score_st = metrics.accuracy_score(y_eval_st, pred_st)
    print("accuracy: {}".format(score_st))

    # Classification Report
    print(classification_report(y_eval_st, pred_st.argmax(-1)))

    
if __name__ == "__main__":
    main()   