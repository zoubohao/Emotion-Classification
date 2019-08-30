import re
import jieba

inputFilePath = "./Data/weibo.txt"
outputFilePath = "./Data/WashedData.txt"


jieba.add_word("蔡徐坤")
jieba.add_word("秒拍视频")
jieba.add_word("网页链接")
jieba.add_word("秒拍")
jieba.add_word("蹦蹦床")
jieba.add_word("冯提莫")


def stopwordslist(filepath):
    stopwords = [thisline.strip() for thisline in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


stopWordsList = stopwordslist(".\\Data\\stopWords.txt")
with open(inputFilePath,'r',encoding="UTF-8") as rh :
    with open(outputFilePath,"w",encoding="UTF-8") as wh:
        for line in rh:
            oneLine = line.strip()
            meData_label = oneLine.split("\t")
            meData = meData_label[0]
            meDataF = meData.replace("(ˊvˋ*)"," ").replace("● ° u ° ●"," ").replace("【","(").replace("】",")").replace("？"," ").replace("。",".")\
            .replace("，",",").replace("！",".")\
            .replace(".",".").replace("?"," ").replace("#"," ").replace("《","(").replace("|"," ").replace(".",".").replace("@"," ").replace("\""," ")\
            .replace("“","\"").replace("”","\"").replace("<","(").replace(">",")").replace("~"," ").replace("（","(").replace("）", ")").replace("》",")")\
            .replace("…",".").replace("：",":").replace(":",":").replace("/"," ").replace("「","").replace("」"," ").replace("*"," ").replace("『"," ")\
            .replace("』"," ").replace("、"," ").replace("emmm"," ").replace("～"," ").replace("ˉ", " ").replace("："," ").replace("／"," ")\
            .replace("(","(").replace(")",")").replace("ヽ"," ").replace("○"," ").replace("^"," ").replace("-"," ").replace(";",",").replace("&",",")\
            .replace("[","(").replace("]",")").replace("♀"," ").replace("ummmmmm"," ").replace("﹃"," ").replace("Zz"," ").replace("-"," ").replace("σ"," ")\
            .replace("′"," ").replace("`"," ").replace("+"," ").replace("φ"," ").replace("ω"," ").replace("→"," ").replace(",",",").replace("☆"," ").replace("·"," ")\
            .replace("；",",").replace("_"," ").replace("＾"," ").replace("０"," ").replace("!",".").replace("▼"," ").replace("★"," ").replace("℃"," ")\
            .replace("mm"," ").replace("￥"," ").replace("zz"," ").replace("%"," ").replace("x"," ").replace("’"," ").replace("［","(").replace("］",")")\
            .replace("—"," ").replace("③"," ").replace("▽"," ").replace("↓"," ").replace("‘","\"").replace("▽"," ").replace("\\"," ").replace("≧"," ")\
            .replace("≦"," ").replace("④"," ").replace("①"," ").replace("②"," ").replace("⑤"," ").replace("⑥"," ").replace("⑦"," ").replace("丶"," ")\
            .replace("●"," ").replace("°"," ").replace("〖","(").replace("〗",")").lower()
            #print(meDataF)
            splitSentence = list(jieba.cut(meDataF))
            if len(meData_label) > 1:
                for word in splitSentence:
                    if re.match(r"\S", word) is not None:
                        if word not in stopWordsList:
                            wh.write(word + " ")
                wh.write("\t" +  str(meData_label[1]) + "\n")









