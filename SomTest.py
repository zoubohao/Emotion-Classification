import tensorflow as tf
from tensorflow import keras

### return h, [h, c]
### batch size is 3, embedding dim is 100, output shape is [3,20]


# ### cell
# lstm = keras.layers.LSTMCell(20,dropout=0.2,recurrent_dropout=0.2)
# testInput = tf.ones(shape=[3,100])
# testInitState = lstm.get_initial_state(batch_size=3,dtype=tf.float32)
# result = lstm(testInput,states = testInitState,training = False)
# print(result)
#
# ### stacked cells to one RNN
# testStackInput = tf.ones(shape=[3,4,100])
# lstmStack = keras.layers.RNN([keras.layers.LSTMCell(20,dropout=0.2,recurrent_dropout=0.2) for _ in range(3)],return_sequences=True)
# testStackState = lstmStack.get_initial_state(inputs=testStackInput)
# resultS = lstmStack(testStackInput,training = False,initial_state=testStackState)
# print(resultS.shape)
#
# ### Bi lstm
# forLSTM = keras.layers.RNN([keras.layers.LSTMCell(20,dropout=0.2,recurrent_dropout=0.2) for _ in range(3)])
# backLSTM = keras.layers.RNN([keras.layers.LSTMCell(20,dropout=0.2,recurrent_dropout=0.2) for _ in range(3)],go_backwards=True)
# resultFor = forLSTM(testStackInput,training = False ,initial_state= forLSTM.get_initial_state(inputs=testStackInput))
# resultBack = backLSTM(testStackInput,training = False ,initial_state= backLSTM.get_initial_state(inputs=testStackInput))
# print(resultFor.shape)
# print(resultBack.shape)
#
#
# testStackInput = tf.ones(shape=[4,100])
# stackedCell = keras.layers.StackedRNNCells([keras.layers.LSTMCell(20,dropout=0.2,recurrent_dropout=0.2) for _ in range(3)])
# print(stackedCell.get_initial_state(inputs=testStackInput))
# result = stackedCell(testStackInput,training = False,states = stackedCell.get_initial_state(inputs=testStackInput))
# print(result[0].shape)
# print(result[1])


# testInput = tf.ones(shape=[3,2,4])
# liner = keras.layers.Dense(5)
# print(liner(testInput))

# test = [1,2,4,5]
# print(test + [0,0,0])


# thisMap = {}
# with open("d:\\DigKeyEnglish2ChineseAAA.txt","r") as rh :
#     for line in rh:
#         oneLine = line.strip()
#         splitWords = oneLine.split("\t")
#         print(oneLine)
#         thisMap[splitWords[0]] = splitWords[1]
#
#
# with open("d:\\DigKeyEnglish2Chinese.txt","w") as wh:
#     for key,value in thisMap.items():
#         wh.write(key + "\t" + value + "\n")

# import matplotlib.pyplot as plt
# numberList = []
# with open("./Data/WashedData.txt" , "r",encoding="utf-8") as rh:
#     for line in rh:
#         numberList.append(len(line.strip().split("\t")[0].split()))
#         print(len(line.strip().split("\t")[0].split()))
#
#
# plt.xlabel("The words contain in one sentence")
# plt.ylabel("The frequency")
# plt.title("The distribution of words in one sentence")
# plt.hist(numberList,100)
# plt.show()

# testInput = tf.ones(shape=[3,10],dtype=tf.int64)
# print(keras.layers.Flatten()(testInput).shape)

# import numpy as np
# test = [1,2,3,4,5,6,7,8]
# testa = np.zeros(shape=[8])
# testb = np.zeros(shape=[8])
# test = np.array(test)
# testa[np.where(test >= 4)] = 1
#
# print(testa + testb)

# import math
# import numpy as np
# positionWeightMatrix = []
# for pos in range(200):
#     tempPE = []
#     for dim in range(256):
#         if dim % 2 == 0 :
#             tempPE.append(math.sin(pos / math.pow(10000, 2. * dim / 64.)))
#         else:
#             tempPE.append(math.cos(pos / math.pow(10000, 2. * dim / 64.)))
#     positionWeightMatrix.append(tempPE)
#
# positionWeightMatrix = np.array(positionWeightMatrix,dtype=np.float32)
# print(positionWeightMatrix)
# print(positionWeightMatrix.shape)
# import tensorflow as tf
#
# print(tf.range(256))

#
# import matplotlib.pyplot as plt
#
# acc = [0.66,0.6850,0.6932,0.70,0.7053,0.7089]
# val_acc = [0.6879,0.7058,0.6972,0.7260,0.7143,0.7231]
# loss = [0.68,0.6531,0.6321,0.6175,0.6021,0.5950]
# val_loss = [0.6584,0.6038,0.5875,0.5903,0.5732,0.5685]
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()


a = [1,2]
for i in range(len(a)):
    if i != len(a)-1:
        print(i)
    else:
        print("asdf")
        print(i)
