import keras
from keras import layers,models,datasets


from keras.datasets import mnist

(x_tr,y_tr,x_te,y_te)=mnist.load_data()
x_tr,x_te=x_tr/255.0,x_te/255.0
x_tr_cnn=x_tr.reshape(-1,28,28,1)
x_te_cnn=x_te.reshape(-1,28,28,1)
ml=models.Sequential()
ml.add(layers.Flatten(input_shape=(28,28)))
ml.add(layers.Dense(128,activation='relu'))
ml.add(layers.Dense(10,activation='softmax'))
ml.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
ml_h=ml.fit(x_tr,y_tr,epochs=10,validation_data=(x_te,y_te))
cm=models.Sequential()
cm.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
cm.add(layers.MaxPooling2D((2,2)))
cm.add(layers.Conv2D(64,(3,3),activation='relu'))
cm.add(layers.MaxPooling2D((2,2)))
cm.add(layers.Flatten())
cm.add(layers.Dense(128,activation='relu'))
cm.add(layers.Dense(10,activation='softmax'))
cm.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
cm_h=cm.fit(x_tr_cnn,y_tr,epochs=10,validation_data=(x_te_cnn,y_te))
ml_acc=ml_h.history['val_accuracy'][-1]
cm_acc=cm_h.history['val_accuracy'][-1]
print("Mlp:",ml_acc)
print("Cnn:",cm_acc)
