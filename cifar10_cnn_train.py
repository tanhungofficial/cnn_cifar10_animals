#***************************************************************
#------  NGUYEN TAN HUNG    MSSV: 16141175  --------------------
#------  NGUYEN MINH TUAN   MSSV: 16141330  --------------------
#------  PHAM VAN LONG      MSSV: 16141194  --------------------
#------  DAO VAN BANG       MSSV: 16141113  --------------------
#***************************************************************

import  os
import time
import itertools
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing import utils
from keras.optimizers import Adam,SGD
import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn import  metrics

classes = []
traindata = []
trainlabel = []
testdata = []
testlabel = []
#load training image
LDR = os.listdir('train_data')
label =0
for ldr in LDR:
    classes.append(ldr.upper())
    path = os.path.join('train_data',ldr)
    LDR_IMG = os.listdir(path)
    for ldr_img in LDR_IMG:
        try:
            path_img = os.path.join(path,ldr_img)
            img = cv2.imread(path_img,0)
            img = img.reshape(32, 32, 1)
            traindata.append(img)
            trainlabel.append(label)
        except:
            pass
    label += 1
#load test image
LDR = os.listdir('test_data')
label =0
for ldr in LDR:
    path = os.path.join('test_data',ldr)
    LDR_IMG = os.listdir(path)
    for ldr_img in LDR_IMG:
        try:
            path_img = os.path.join(path,ldr_img)
            img = cv2.imread(path_img,0)
            img = img.reshape(32,32,1)
            testdata.append(img)
            testlabel.append(label)
        except:
            pass
    label += 1


#normalize data
num_classes = len(classes)
trainlabel = np.array(trainlabel)
traindata = np.array(traindata)/255
testdata = np.array(testdata)/255
testlabel = np.array(testlabel)
trainlabel = utils.to_categorical(trainlabel)
testlabel = utils.to_categorical(testlabel)
'''
#save data is normalized
data_dict = {'traindata': traindata,'trainlabel':trainlabel,'testdata':testdata,'testlabel':testlabel,'classes':classes}
with open('cifar10_data','wb') as pickle_out:
    pickle.dump(data_dict,pickle_out)
'''
with open("file_name",'wb') as pickle_out:
    pickle.dump(classes,pickle_out)
'''
# Load saved data for later, it is faster
data_dict=[]
with open("cifar10_data",'rb') as pickle_in:
    data_dict = pickle.load(pickle_in)
    print(data_dict.keys())
traindata = data_dict['traindata']
trainlabel = data_dict['trainlabel']
testlabel = data_dict['testlabel']
testdata = data_dict['testdata']
classes = data_dict['classes']
num_classes = len(classes)
'''

#Built model
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = (32,32,1), activation = 'relu'))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = num_classes, activation = 'softmax'))

# initialize our initial learning rate and # of epochs to train for
learning_rate = 0.0001
EPOCHS = 100
batch_size = 100

# Compiling the CNN
opt = Adam(learning_rate=learning_rate)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()
#Training the CNN to the images
start_time = time.time()
H = model.fit(traindata, trainlabel, validation_data=(testdata, testlabel), epochs=EPOCHS, batch_size=batch_size)
end_time = time.time()
tot_time = end_time - start_time
print("\nTotal Elapsed Runtime:", tot_time, "in seconds.")
print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) )+ ":" +
	str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":"+
	str( int(  ( (tot_time % 3600) % 60 ) ) ))
# save the model
model.save('cifar10_cnn.model')

# save training history 
with open("training_history",'wb') as pickle_out:
    pickle.dump(H,pickle_out)
'''
with open("training_history",'rb') as pickle_in:
    H = pickle.load(pickle_in)
'''
# evaluate the network
model = keras.models.load_model('cifar10_cnn.model')
print("[INFO] evaluating network...")
predictions = model.predict(testdata, batch_size=32)
print(classification_report(testlabel.argmax(axis=1),
    predictions.argmax(axis=1)))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("plot")

#create confustion matrix
cnf_matrix = metrics.confusion_matrix(np.argmax(testlabel,axis=1), np.argmax(predictions,axis=1))
print(cnf_matrix/np.sum(cnf_matrix,axis=1,keepdims=True))
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims =
        True)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),
    range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix")

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,normalize=True,title='Normalized confusion matrix')
plt.show()
