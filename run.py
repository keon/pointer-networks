from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from PointerLSTM import PointerLSTM
import pickle
import tsp_data as tsp
import numpy as np
import keras


def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1

print("preparing dataset...")
t = tsp.Tsp()
X, Y = t.next_batch(10000)
x_test, y_test = t.next_batch(100)

YY = []
for y in Y:
    YY.append(to_categorical(y))
YY = np.asarray(YY)

hidden_size = 128
seq_len = 10
nb_epochs = 10000
learning_rate = 0.1

print("building model...")
main_input = Input(shape=(seq_len, 2), name='main_input')

encoder,state_h, state_c = LSTM(hidden_size,return_sequences = True, name="encoder",return_state=True)(main_input)
decoder = PointerLSTM(hidden_size, name="decoder")(encoder,states=[state_h, state_c])

model = Model(main_input, decoder)
print(model.summary())
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, YY, epochs=nb_epochs, batch_size=64,)
print(model.predict(x_test))
print('evaluate : ',model.evaluate(x_test,to_categorical(y_test)))
print("------")
print(to_categorical(y_test))
model.save_weights('model_weight_100.hdf5')
