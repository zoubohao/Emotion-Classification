import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras import backend as K

l2Lambda = 0



class SelfAttention(keras.Model):

    def __init__(self,dk):
        super().__init__()
        self.dk = tf.convert_to_tensor(dk,dtype=tf.float32)
        self.dropout = keras.layers.Dropout(rate=0.1)

    def call(self, inputs, training=None, mask=None):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]
        qkTensor = K.math_ops.matmul(q,k,transpose_b=True)
        scaleTensor = K.math_ops.multiply(K.stop_gradient(1. / K.math_ops.sqrt(self.dk)) , qkTensor)
        softMaxTensor =  K.softmax(scaleTensor)
        drT = self.dropout(softMaxTensor,training = training)
        vTensor = K.math_ops.matmul(drT,v)
        return vTensor



class Multi_Head_Attention(keras.Model):

    def __init__(self,units,numberOfBlocks):
        super().__init__()
        self.numberB = numberOfBlocks
        self.linerQ = [keras.layers.Dense(units // numberOfBlocks) for _ in range(numberOfBlocks)]
        self.linerK = [keras.layers.Dense(units // numberOfBlocks) for _ in range(numberOfBlocks)]
        self.linerV = [keras.layers.Dense(units // numberOfBlocks) for _ in range(numberOfBlocks)]
        self.attention = [SelfAttention(units) for _ in range(numberOfBlocks)]
        self.outLiner = keras.layers.Dense(units)
        self.ln = keras.layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        QTensor = inputs[0]
        KTensor = inputs[1]
        VTensor = inputs[2]
        concatList = []
        for i in range(self.numberB):
            thisQTrans = self.linerQ[i](QTensor)

            thisVTrans = self.linerV[i](VTensor)

            thisKTrans = self.linerK[i](KTensor)

            thisAttentionT =self.attention[i]([thisQTrans,thisVTrans,thisKTrans],training = training)
            concatList.append(thisAttentionT)
        concatTensor = tf.concat(concatList,axis=-1)
        outTensor = self.outLiner(concatTensor)
        addTensor = K.math_ops.add(outTensor,QTensor)
        lnT = self.ln(addTensor)
        return lnT


class FeedForward(keras.Model) :

    def __init__(self,outputDim):
        super(FeedForward,self).__init__()
        self.dense0 = keras.layers.Dense(outputDim,activation=keras.activations.relu)
        self.dense2 = keras.layers.Dense(outputDim)
        self.ln = keras.layers.LayerNormalization()


    def call(self, inputs, training=None, mask=None):
        d0 = self.dense0(inputs)
        d2 = self.dense2(d0)
        addTensor = K.math_ops.add(d2,inputs)
        return self.ln(addTensor)

class TransformerEncoder(keras.Model):

    def __init__(self,units):
        super().__init__()
        self.multiHead = Multi_Head_Attention(units=units,numberOfBlocks=8)
        self.feedForward = FeedForward(outputDim=units)

    def call(self, inputs, training=None, mask=None):
        multiTensor = self.multiHead([inputs,inputs,inputs],training = training)
        feedTensor = self.feedForward(multiTensor,training = training)
        return feedTensor



class Transformer(keras.Model):

    def __init__(self,embeddingMatrix,maxWordsInOneSentence,embeddingSize,labelsNumber,units = 512 ,numberOfTransformer = 6):
        self._transformerLayers = numberOfTransformer
        super().__init__()
        self.dModel = tf.convert_to_tensor(units,dtype=tf.float32)
        self.iniEmbeddingMatrix  = tf.convert_to_tensor(embeddingMatrix,dtype=tf.float32)
        self.positionEmbeddingMatrix = keras.layers.Embedding(input_dim=maxWordsInOneSentence + 1,output_dim=embeddingSize)
        self.embeddingDrop = keras.layers.Dropout(rate=0.1)

        self.denseTrans = keras.layers.Dense(units)

        self.transformerList = [TransformerEncoder(units=units) for _ in range(numberOfTransformer)]

        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(labelsNumber)


    ### [batch , times ]
    def call(self, inputs, training=None, mask=None):
        sentence = inputs[0]
        position = inputs[1]
        batchTensor = tf.stop_gradient(tf.nn.embedding_lookup(params=self.iniEmbeddingMatrix, ids=sentence))
        batchTensor = K.stop_gradient(K.math_ops.multiply(batchTensor,K.math_ops.sqrt(self.dModel)))

        positionTensor = self.positionEmbeddingMatrix(position)
        positionTensor = K.math_ops.multiply(positionTensor,K.stop_gradient(K.math_ops.sqrt(self.dModel)))
        # print(batchTensor.shape)
        # print(positionTensor.shape)

        eDropTensor = self.embeddingDrop(K.math_ops.multiply(K.math_ops.add(batchTensor,positionTensor),
                                                             K.stop_gradient(tf.convert_to_tensor(1. / 2.,dtype=tf.float32))),
                                         training = training)
        denseTrans = self.denseTrans(eDropTensor)

        thisTransformer = K.identity(denseTrans)
        for i in range(self._transformerLayers):
            thisTransformer = self.transformerList[i](thisTransformer,training=training)

        flattenTensor = self.flat(thisTransformer)
        dense1Tensor = self.dense1(flattenTensor)
        return tf.nn.softmax(dense1Tensor,axis=-1)



if __name__ == "__main__":
    testInput = tf.ones(shape=[3,10],dtype=tf.int64)
    model = Transformer(np.ones(shape=[100,150],dtype=np.float32),labelsNumber=2,embeddingSize=150,maxWordsInOneSentence=10)
    result = model([testInput,testInput],training = False)
    print(result)
    # testInput1 = tf.ones(shape=[3, 10, 512], dtype=tf.float32)
    # s = SelfAttention(512)
    # print(s([testInput1,testInput1,testInput1]).shape)


