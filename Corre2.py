def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False



# print("Output final file.")
# finalOutPath = "d:\\DigKeyGoodsPair.txt"
# with open("d:\\pariOutput.txt","r",encoding="utf-8") as rh:
#     with open(finalOutPath,"w",encoding="utf-8") as wh:
#         for line in rh:
#             oneLine = line.strip()
#             if is_Chinese(oneLine):
#                 wh.write(oneLine + "\n")





thisMap = {}
with open("d:\\DigKeyGoodsPair.txt","r",encoding="utf-8") as rh:
    for line in rh:
        oneLine = line.strip()
        if "厂方库存" not in oneLine and "非库存货" not in oneLine and "原厂标准交货期" not in oneLine and "立即发货" not in oneLine \
                and "宽 x" not in oneLine and "长 x" not in oneLine:
            thisMap[oneLine.split("\t")[0]] = oneLine.split("\t")[1]

with open("d:\\FinalDigKeyGoodsPair.txt","w",encoding="utf-8") as wh :
    for key ,value in thisMap.items():
        if value != "-":
            wh.write(key + "\t" + value + "\n")



