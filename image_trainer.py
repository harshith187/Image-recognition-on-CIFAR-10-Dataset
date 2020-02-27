from keras.datasets import cifar10
from keras.models import  Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import MaxPooling2D,Conv2D
from keras.optimizers import SGD
from keras.utils import np_utils

(X_train,y_train),(X_test,y_test) = cifar10.load_data()
new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')
new_X_test /= 255
new_X_train /= 255
new_Y_train = np_utils.to_categorical(y_train)
new_Y_test = np_utils.to_categorical(y_test)

lables = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

model = Sequential()
model.add(Conv2D(32, kernel_size= 2, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, kernel_size= 2, activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, kernel_size= 2, activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
training = model.fit(new_X_train,new_Y_train,epochs=10,batch_size=32,validation_split=.2)
model.save('Trained_model.h5')