import tensorflow as tf
import math
import numpy as np
from C_LSTM import LSTM_Model
from C_BiLSTM import Bi_LSTM_Model
from C_BiLSTM_Attention import Bi_LSTM_Attention
from C_Transformer import Transformer
import sklearn.metrics as metrics
from tensorflow  import keras


###### Data config
weightMatrixPath = "./Data/weightMatrix.txt"
oneLabelDataPath = "./Data/oneData.txt"
zeroLabelDataPath = "./Data/zeroData.txt"
embeddingSize = 256
batchSize = 64
maxWordsInOneSentence = 200
trainTestSplitRatio = 0.99
###### Training config
trainOrTest = "Train"
lr = 1e-5
l2Lambada = 5e-5
epoch = 25
trainingTimesInOneEpoch = 2400
disPlayTimes = 50
decayRate = 0.98
decayTimes = 2399
saveModelTimes = 2399
saveParamPath = ".\\Transformer_Model_"
trainingLoadWeight = False
trainingLoadWeightPath = ".\\Transformer_Model_9"
####### Test config
testModeWeight = ".\\Transformer_Model_10"


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
print("The shape of weight matrix is :",weightMatrix.shape)

############################
### Change model at here ###

# model = LSTM_Model(embeddingMatrix=weightMatrix,labelNumbers=labelsNumber)
# model = Bi_LSTM_Model(embeddingMatrix=weightMatrix,labelNumbers=labelsNumber)
# model = Bi_LSTM_Attention(embeddingMatrix=weightMatrix,labelsNumber=labelsNumber)
model = Transformer(embeddingMatrix=weightMatrix,labelsNumber=1)

### Change model at here ###
############################


print("Loading Data.")
oneLabelData = []
with open(oneLabelDataPath,"r",encoding="UTF-8") as rh:
    for line in rh:
        oneLine = line.strip()
        wordList = oneLine.split()
        thisSentence = []
        for word in wordList:
            idx = vocabulary2idx[word]
            thisSentence.append(idx)
            if len(thisSentence) == maxWordsInOneSentence :
                break
        if len(thisSentence) < maxWordsInOneSentence:
            paddingZeros = maxWordsInOneSentence - len(thisSentence)
            oneLabelData.append(thisSentence + [0 for _ in range(paddingZeros)])
        else:
            oneLabelData.append(thisSentence)

zeroLabelData = []
with open(zeroLabelDataPath,"r",encoding="UTF-8") as rh:
    for line in rh:
        oneLine = line.strip()
        wordList = oneLine.split()
        thisSentence = []
        for word in wordList:
            idx = vocabulary2idx[word]
            thisSentence.append(idx)
            if len(thisSentence) == maxWordsInOneSentence :
                break
        if len(thisSentence) < maxWordsInOneSentence:
            paddingZeros = maxWordsInOneSentence - len(thisSentence)
            zeroLabelData.append(thisSentence + [0 for _ in range(paddingZeros)])
        else:
            zeroLabelData.append(thisSentence)
print("Loading data completed.")
oneLabelData = np.array(oneLabelData,dtype=np.int64)
zeroLabelData = np.array(zeroLabelData,dtype=np.int64)
print("There are " + str(oneLabelData.shape) + " in one label file.")
print("There are " + str(zeroLabelData.shape) + " in zero label file.")

oneLabelTrainData = oneLabelData[0:int(oneLabelData.shape[0] * trainTestSplitRatio),:]
oneLabelTestData = oneLabelData[int(oneLabelData.shape[0] * trainTestSplitRatio):,:]

zeroLabelTrainData = zeroLabelData[0:int(zeroLabelData.shape[0] * trainTestSplitRatio),:]
zeroLabelTestData = zeroLabelData[int(zeroLabelData.shape[0] * trainTestSplitRatio):,:]

def DataGenerator(dataNumpy,labelType):
    while True:
        for oneData in dataNumpy:
            yield np.array(oneData, dtype=np.int64), labelType

oneDataGenerator = DataGenerator(oneLabelTrainData,1)
zeroDataGenerator = DataGenerator(zeroLabelTrainData,0)

# oneBatchData = []
# oneBatchLabel = []
# for b in range(batchSize // 2):
#     thisOneData, thisOneLabel = oneDataGenerator.__next__()
#     thisZeroData,thisZeroLabel = zeroDataGenerator.__next__()
#     oneBatchData.append(thisOneData)
#     oneBatchData.append(thisZeroData)
#     oneBatchLabel.append(thisOneLabel)
#     oneBatchLabel.append(thisZeroLabel)
# print(oneBatchData)
# print(oneBatchLabel)
# print(weightMatrix[2108])
# print(weightMatrix[0])


optimizer = tf.optimizers.Adam(learning_rate=lr)


@tf.function
def train(batchData, labels):
    with tf.GradientTape() as tape:
        predict = model(batchData, training=True)
        # print(predict.shape)
        loss = keras.losses.binary_crossentropy(y_true=labels,y_pred=tf.squeeze(predict),from_logits=True)
        l2Tensors = []
        for var in model.trainable_variables:
            name = var.name
            if "batch" not in name and "bias" not in name:
                l2Tensors.append(tf.nn.l2_loss(var))
        l2Loss = tf.multiply(l2Lambada, tf.add_n(l2Tensors))
        totalLoss = loss + l2Loss
        # print(totalLoss)
        gradients = tape.gradient(target=totalLoss, sources=model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return predict, float(loss), float(totalLoss)

if trainOrTest.lower() == "train":
    trainingTimes = 0
    if trainingLoadWeight :
        testInput = tf.zeros(shape=[batchSize, maxWordsInOneSentence], dtype=tf.int64)
        model(testInput, training=False)
        model.load_weights(trainingLoadWeightPath)
    for e in range(epoch):
        for times in range(trainingTimesInOneEpoch):
            oneBatchData = []
            oneBatchLabel = []
            for b in range(batchSize // 2):
                thisOneData, thisOneLabel = oneDataGenerator.__next__()
                thisZeroData, thisZeroLabel = zeroDataGenerator.__next__()
                oneBatchData.append(thisOneData)
                oneBatchData.append(thisZeroData)
                oneBatchLabel.append(float(thisOneLabel))
                oneBatchLabel.append(float(thisZeroLabel))
            oneBatchArray = np.array(oneBatchData, dtype=np.int64)
            oneBatchLabelArray = np.array(oneBatchLabel, dtype=np.float32)
            #print(oneBatchArray.shape)
            predictLogits, labelsLoss, totalLosses = train(oneBatchArray, oneBatchLabelArray)
            if trainingTimes % disPlayTimes == 0:
                print("#################")
                print("Training times :", trainingTimes)
                print("Predict logits : ", np.array(tf.squeeze(tf.sigmoid(predictLogits)))[0:4])
                print("Truth labels ", oneBatchLabelArray[0:4])
                print("Label loss is ", labelsLoss)
                print("Total loss is ", totalLosses)
                config = optimizer.get_config()
                print("Learning rate is ", config["learning_rate"])
            trainingTimes = trainingTimes + 1
            if trainingTimes % decayTimes == 0 and trainingTimes != 0:
                lr = lr * math.pow(decayRate, trainingTimes / decayTimes + 0.0)
                ###
                config = optimizer.get_config()
                config["learning_rate"] = lr
                optimizer = optimizer.from_config(config)
            if trainingTimes % saveModelTimes == 0 and trainingTimes != 0:
                print("Saving Model.")
                model.save_weights(filepath=saveParamPath + str(e))
                print("Saving complete.")
else:
    predictLabels = []
    truthLabels = []
    testInput = tf.zeros(shape=[1,maxWordsInOneSentence],dtype=tf.int64)
    model(testInput,training = False)
    model.load_weights(testModeWeight)
    print("There are " + str(len(oneLabelTestData)) + " one label test data .")
    k = 0
    for thisData in oneLabelTestData:
        #print(thisData.shape)
        thisDataRe = np.reshape(thisData,newshape=[1,maxWordsInOneSentence])
        #print(thisDataRe.shape)
        onePredict = model(thisDataRe,training = False)
        preLocation = np.argmax(tf.squeeze(onePredict))
        predictLabels.append(preLocation)
        truthLabels.append(1)
        print("Predict is " + str(preLocation) + " , Truth is " + str(1) + " " + str(k))
        k += 1
    print("There are " + str(len(zeroLabelTestData)) + " zero label test data .")
    k = 0
    for thisData in zeroLabelTestData:
        thisDataRe = np.reshape(thisData,newshape=[1,maxWordsInOneSentence])
        onePredict = model(thisDataRe,training = False)
        preLocation = np.argmax(tf.squeeze(onePredict))
        predictLabels.append(preLocation)
        truthLabels.append(0)
        print("Predict is " + str(preLocation) + " , Truth is " + str(0) + " " + str(k))
        k+=1
    recall = metrics.recall_score(truthLabels,predictLabels)
    precision = metrics.precision_score(truthLabels,predictLabels)
    microF1 = metrics.f1_score(y_pred=predictLabels, y_true=truthLabels, average='micro', labels=[0,1])
    macroF1 = metrics.f1_score(truthLabels, predictLabels, average="macro", labels=[0,1])
    eachLabelF1 = metrics.f1_score(truthLabels, predictLabels, average=None)
    acc = metrics.accuracy_score(y_true=truthLabels,y_pred=predictLabels)
    print("Recall is ",recall)
    print("Precision is ",precision)
    print("Micro F1 is ",microF1)
    print("Macro F1 is ",macroF1)
    print("Each label F1 is ",eachLabelF1)
    print("Accuracy is ",acc)








