import numpy as np
from C_LSTM import LSTM_Model
from C_BiLSTM import Bi_LSTM_Model
from C_BiLSTM_Attention import Bi_LSTM_Attention
from C_Transformer import Transformer
import sklearn.metrics as metrics
from tensorflow  import keras
import  re
import tensorflow as tf


###### Data config
weightMatrixPath = "./Data/weightMatrix.txt"
dataPath = "./Data/WashedData.txt"
embeddingSize = 256
batchSize = 64
maxWordsInOneSentence = 100
testSamplesNumber = 1000
###### Training config
trainingOrTesting = "Train"
epoch = 40

saveParamPath = ".\\Transformer\\Transformer"
####### Test config
testModeWeight = ".\\Transformer\\Transformer_0.702"


print("Loading weight matrix.")
vocabulary2idx = {}
weightMatrix = [np.zeros(shape=[embeddingSize],dtype=np.float32)]
with open(weightMatrixPath,"r",encoding="UTF-8") as wh:
    for i , line in enumerate(wh):
        oneLine = line.strip()
        word_vec = oneLine.split("\t")
        word = word_vec[0]
        vec = word_vec[1]
        vecS = vec.split(",")[0:-1]
        vocabulary2idx[word] = i + 1
        thisVec = []
        for num in vecS:
            thisVec.append(float(num))
        weightMatrix.append(np.array(thisVec,dtype=np.float32))
weightMatrix = np.array(weightMatrix,dtype=np.float32)
print("Loading completed.")
print("There are " + str(len(vocabulary2idx)) + " vocabularies in this file.")


############################
### Change model at here ###

#model = LSTM_Model(embeddingMatrix=weightMatrix,labelNumbers=2)
#model = Bi_LSTM_Model(embeddingMatrix=weightMatrix,labelNumbers=2)
#model = Bi_LSTM_Attention(embeddingMatrix=weightMatrix,labelsNumber=2)
model = Transformer(embeddingMatrix=weightMatrix,maxWordsInOneSentence=maxWordsInOneSentence,embeddingSize=embeddingSize,labelsNumber=2)

### Change model at here ###
############################



print("Loading Data.")
data = []
labels = []
positions = []
with open(dataPath,"r",encoding="UTF-8") as rh:
    for line in rh:
        oneLine = line.strip()
        sentence_labels = oneLine.split("\t")
        if len(sentence_labels) > 1:
            if re.fullmatch(r"[01]", sentence_labels[1]) is not None:
                position = 1
                wordList = sentence_labels[0].split()
                thisSentence = []
                thisZeroNp = np.zeros(shape=[2],dtype=np.float32)
                thisZeroNp[int(sentence_labels[1])] = 1.
                labels.append(thisZeroNp)
                thisPosition = []
                #print(float(sentence_labels[1]))
                for word in wordList:
                    idx = vocabulary2idx[word]
                    thisSentence.append(idx)
                    thisPosition.append(position)
                    position += 1
                    if len(thisSentence) == maxWordsInOneSentence:
                        break
                if len(thisSentence) < maxWordsInOneSentence:
                    paddingZeros = maxWordsInOneSentence - len(thisSentence)
                    data.append(np.array(thisSentence + [0 for _ in range(paddingZeros)],dtype=np.int64))
                    positions.append(np.array(thisPosition + [0 for _ in range(paddingZeros)],dtype=np.int64))
                else:
                    data.append(np.array(thisSentence, dtype=np.int64))
                    positions.append(np.array(thisPosition,dtype=np.int64))
lenData = len(data)
print(lenData)
data = np.array(data,dtype=np.int64)
labels = np.array(labels, dtype=np.float32)
positions = np.array(positions,dtype=np.int64)
TrainData = []
TrainLabels = []
TrainPosition = []
TestOneData = []
TestOneLabels = []
TestOnePosition = []
TestZeroData = []
TestZeroLabels = []
TestZeroPosition = []
for i,thisOneData in enumerate(data):
    if np.argmax(labels[i]) == 0:
        TestZeroData.append(thisOneData)
        TestZeroLabels.append(labels[i])
        TestZeroPosition.append(positions[i])
        if len(TestZeroData) > testSamplesNumber // 2:
            TestZeroData.pop(-1)
            TestZeroLabels.pop(-1)
            TestZeroPosition.pop(-1)
            TrainData.append(thisOneData)
            TrainLabels.append(labels[i])
            TrainPosition.append(positions[i])
    else:
        TestOneData.append(thisOneData)
        TestOneLabels.append(labels[i])
        TestOnePosition.append(positions[i])
        if len(TestOneData) > testSamplesNumber // 2:
            TestOneData.pop(-1)
            TestOneLabels.pop(-1)
            TestOnePosition.pop(-1)
            TrainData.append(thisOneData)
            TrainLabels.append(labels[i])
            TrainPosition.append(positions[i])

# for lab in TrainLabels:
#     print(lab)

TestPosition = TestOnePosition + TestZeroPosition
TestData = TestOneData + TestZeroData
TestLabels = TestOneLabels + TestZeroLabels

TrainData = np.array(TrainData)
TrainPosition = np.array(TrainPosition)
TrainLabels = np.array(TrainLabels)
TestData = np.array(TestData)
TestPosition = np.array(TestPosition)
TestLabels = np.array(TestLabels)
print("Training data.")
print(TrainData)
print(TrainData.shape)
print("Training labels.")
print(TrainLabels)
print(TrainPosition)
print(TrainPosition[0])
print(np.max(TrainPosition,axis=-1))
print("Test data.")
print(TestData)
print(TestData.shape)
print("Test labels.")
print(TestLabels)

print(TrainPosition)


print("The shape of weight matrix is :",weightMatrix.shape)
sampleWeight = []
for oneLabel in TrainLabels:
    if np.argmax(oneLabel) == 0.:
        sampleWeight.append(0.62)
    else:
        sampleWeight.append(1.0)
sampleWeight = np.array(sampleWeight,dtype=np.float32)
print(sampleWeight)


model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-5,momentum=0.9,nesterov=True),
                  loss=tf.losses.CategoricalCrossentropy(label_smoothing=0.001),
                  metrics=['acc'])
history = model.fit([TrainData, TrainPosition], TrainLabels,
                    epochs=epoch,
                    shuffle=True,
                    batch_size=batchSize,
                    validation_data=([TestData, TestPosition], TestLabels),
                    class_weight={0: 0.78, 1: 1.0},
                    sample_weight=sampleWeight,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_acc",patience=10, restore_best_weights=True),
                               keras.callbacks.TensorBoard(),
                               keras.callbacks.ReduceLROnPlateau(monitor="val_acc",factor=0.1,patience=3,min_lr=1e-12,verbose=1),
                               keras.callbacks.BaseLogger(),
                               keras.callbacks.CSVLogger(".\\training.csv")])
model.save_weights(saveParamPath)

# # #'binary_crossentropy'
# if trainingOrTesting.lower() == "train":
#     model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-5,momentum=0.9,nesterov=True),
#                   loss=tf.losses.CategoricalCrossentropy(label_smoothing=0.05),
#                   metrics=['acc'])
#     valACC = 0.
#     acc = []
#     val_acc = []
#     loss = []
#     val_loss = []
#     for e in range(epoch):
#         print("epoch : ",e)
#         if e >= 1:
#             model.load_weights(saveParamPath + "_" + str(valACC))
#         if "Transformer" in saveParamPath:
#             history = model.fit([TrainData,TrainPosition], TrainLabels,
#                                 epochs=1,
#                                 shuffle=True,
#                                 batch_size=batchSize,
#                                 validation_data=([TestData,TestPosition], TestLabels),
#                                 class_weight={0:0.88,1:1.0},
#                                 sample_weight=sampleWeight,
#                                 callbacks=[keras.callbacks.EarlyStopping(monitor="acc",restore_best_weights=True),
#                                            keras.callbacks.TensorBoard(),
#                                            keras.callbacks.LearningRateScheduler(schedule,verbose=1)]
#                                 )
#         else:
#             history = model.fit(TrainData, TrainLabels,
#                                 epochs=1,
#                                 shuffle=True,
#                                 batch_size=batchSize,
#                                 validation_data=(TestData, TestLabels),
#                                 # class_weight={0:0.5885,1:1.0},
#                                 sample_weight=sampleWeight,
#                                 )
#         valACC = history.history["val_acc"][0]
#         ACC = history.history["acc"][0]
#         LOSS = history.history["loss"][0]
#         valLOSS = history.history['val_loss'][0]
#         model.save_weights(saveParamPath + "_" + str(valACC))
#         val_acc.append(valACC)
#         acc.append(ACC)
#         loss.append(LOSS)
#         val_loss.append(valLOSS)
#     import matplotlib.pyplot as plt
#     epochs = range(1, len(acc) + 1)
#
#     plt.plot(epochs, acc, 'bo', label='Training acc')
#     plt.plot(epochs, val_acc, 'b', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#
#     plt.figure()
#
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#
#     plt.show()
#
# else:
#     print("Predict.")
#     model.load_weights(testModeWeight)
#     predictNumpy = model.predict(TestData)
#     predictArg = []
#     truthArg = []
#     for i,thisLabel in enumerate(predictNumpy):
#         predictArg.append(np.argmax(thisLabel))
#         truthArg.append(np.argmax(TestLabels[i]))
#     confusionMatrix = metrics.confusion_matrix(truthArg,predictArg)
#     acc = metrics.accuracy_score(y_true=truthArg,y_pred=predictArg)
#     TN = confusionMatrix[0,0]
#     FN = confusionMatrix[1,0]
#     FP = confusionMatrix[0,1]
#     TP = confusionMatrix[1,1]
#     print([[TP,FN],[FP,TN]])
#     print("Recall is ",TP / (TP + FN) + 0.)
#     print("Precision is ",TP / (TP + FP) + 0.)
#     print("F1 is ",2 * TP / (2 * TP + FN + FP )  + 0.)
#     print("Accuracy is ",acc)
















