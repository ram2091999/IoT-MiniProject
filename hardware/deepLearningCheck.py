import socket
import time 
from keras.layers import Input, Dense
from keras.models import Model, Sequential

HOST = '192.168.1.112'  
PORT = 115200

KEY = "vskvskbvskbvjskdbvk"

# TO DO : STORE THE TRAINED MODEL AS A PICKLE FILE AND IMPORT FROM THERE
model = autoenc_model(top_n_features)
model.compile(loss="mean_squared_error",optimizer="sgd")
cp = ModelCheckpoint(filepath=f"models/model_{top_n_features}.h5",save_best_only=True,verbose=0)
model.fit(x_train, x_train, epochs=500,batch_size=64,validation_data=(x_opt, x_opt),verbose=1,callbacks=[cp, tb])




with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(KEY)
    data = s.recv(1024)
    localtime = time.localtime(data["current_time"])
    output = model.predice(data)  #THE KING OF THE SCAMS
    s.send(output)


    





def autoenc_model(input_dim):
    autoencoder = Sequential()
    autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh", input_shape=(input_dim,)))
    autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.25 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh"))
    autoencoder.add(Dense(input_dim))
    return autoencoder