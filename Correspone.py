import os
import numpy as np


fileFolder = "D:\\DigKey\\"
mapPath = "d:\\DigKeyEnglish2Chinese.txt"
outputFilePath = "d:\\pariOutput.txt"
finalOutPath = "d:\\DigKeyGoodsPair.txt"

english2chinese = {}
chinese2english = {}
with open(mapPath,"r") as rh:
    for line in rh:
        eng = line.strip().split("\t")[1]
        chi = line.strip().split("\t")[0]
        english2chinese[eng] = chi
        chinese2english[chi] = eng

nameSet = set()
for root, folder , files in os.walk(fileFolder):
    for file in files:
        nameSet.add(file.split("_")[0])
print(nameSet)

chineseDataList = []
englishDataList = []
print(len(nameSet))
thisK = 0
for name in nameSet:
    print(thisK)
    print(name)
    chineseName = fileFolder + name + "_zh.txt"
    englishName = fileFolder + name + "_en.txt"
    with open(chineseName,"r",encoding="UTF-8") as rh:
        for oneLine in rh:
            #print(oneLine.strip().split("\t"))
            chineseDataList.append(oneLine.strip().split("\t"))
    with open(englishName,"r",encoding="UTF-8") as rh:
        for oneLine in rh:
            #print(oneLine.strip().split("\t"))
            englishDataList.append(oneLine.strip().split("\t"))
    numberOfRows = len(chineseDataList)
    indexChineseList = []
    indexEnglishList = []
    for i,oneSentence in enumerate(chineseDataList):
        if "比较零件" in oneSentence:
            indexChineseList.append(i)
    for i,oneSentence in enumerate(englishDataList):
        #print(oneSentence)
        if "Compare Parts" in oneSentence:
            indexEnglishList.append(i)
    #print(indexChineseList)
    #print(indexEnglishList)
    for i in range(len(indexChineseList)):
        #print("#############")
        if i != len(indexChineseList)-1:
            oneBatchChinese = chineseDataList[indexChineseList[i]:indexChineseList[i+1]]
            oneBatchEnglish = englishDataList[indexEnglishList[i]:indexEnglishList[i+1]]
        else:
            oneBatchChinese = chineseDataList[indexChineseList[i]:-1]
            oneBatchEnglish = englishDataList[indexEnglishList[i]:-1]
        chineseTitle = oneBatchChinese[0]
        englishTitle = oneBatchEnglish[0]
        #print(chineseTitle)
        #print(englishTitle)
        thisBatchC2EMap = {}
        #print(chinese2english)
        for label in chineseTitle:
            if label in chinese2english:
                englishTrans = chinese2english[label]
                if englishTrans in englishTitle:
                    labelCIndex = chineseTitle.index(label)
                    labelEIndex = englishTitle.index(englishTrans)
                    thisBatchC2EMap[labelCIndex] = labelEIndex
        #print(thisBatchC2EMap)
        chineseId = []
        englishId = []
        for s , oneSentence in enumerate(oneBatchChinese):
            if s != 0:
                chineseId.append(oneSentence[3])
        for s,oneSentence in enumerate(oneBatchEnglish):
            if s != 0:
                englishId.append(oneSentence[3])
        #print(chineseId)
        #print(englishId)
        shareId = []
        for thisId in chineseId:
            if thisId in englishId:
                shareId.append(thisId)
        #print(shareId)
        shareInforChineseData = []
        shareInforEnglishData = []
        for oneSentence in oneBatchChinese:
            if oneSentence[3] in shareId:
                shareInforChineseData.append(oneSentence)
        for oneSentence in oneBatchEnglish:
            if oneSentence[3] in shareId:
                shareInforEnglishData.append(oneSentence)
        #print(len(shareInforChineseData))
        #print(len(shareInforEnglishData))
        pair = {}
        for oneCSentence in shareInforChineseData:
            idNm = oneCSentence[3]
            eSentence = []
            for oneESentence in shareInforEnglishData:
                if oneESentence[3] == idNm:
                    eSentence = shareInforEnglishData[shareInforEnglishData.index(oneESentence)]
                    break
            #print(oneCSentence)
            #print(eSentence)
            for cPosition,ePosition in thisBatchC2EMap.items():
                if cPosition <= len(oneCSentence) - 1 and ePosition <= len(eSentence) - 1:
                    pair[oneCSentence[cPosition]] = eSentence[ePosition]
        #print(pair)
        with open(outputFilePath,"a",encoding="UTF-8") as wh:
            for key ,value in pair.items():
                wh.write(key + "\t" + value + "\n")
    thisK += 1


def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

print("Output final file.")
with open(outputFilePath,"r",encoding="utf-8") as rh:
    with open(finalOutPath,"w",encoding="utf-8") as wh:
        for line in rh:
            oneLine = line.strip()
            if is_Chinese(oneLine):
                if "厂方库存" not in oneLine and "非库存货" not in oneLine and "原厂标准交货期" not in oneLine and "立即发货" not in oneLine \
                        and "宽 x" not in oneLine and "长 x" not in oneLine:
                        wh.write(oneLine + "\n")



















