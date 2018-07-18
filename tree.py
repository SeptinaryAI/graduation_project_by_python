import math
import operator
import sort
import copy

#计算熵值 
#传入参数：计算熵值的数据List
def calcEntropy(dataSet):
    entropy = 0                             #熵值初始化
    num = len(dataSet)                      #数据条数
    labelCounts={}                          #项目字典初始化,格式{'类型1':数量;'类型f':数量}
    for each in dataSet:
        if each[-1] not in labelCounts:     #新的项目加入字典
            labelCounts[each[-1]] = 0       #计数初始为0
        labelCounts[each[-1]] += 1          #存在项目则计数+1
    for key in labelCounts:
        prob = float(labelCounts[key])/num  
        entropy  -= prob*math.log(prob,2)
    return entropy
    
#划分数据
#传入参数：用于划分的数据List、划分特征的列序号、划分特征的值
def splitDataSet(dataSet,axis,value): 
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
#找到最好的划分feature
#传入参数：用于划分的数据List
def chooseBestFeatureToSplit(dataSet):
    featureNum = len(dataSet[0]) - 1                    #feature个数
    baseEntropy = calcEntropy(dataSet)                  #整个dataset的熵
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(featureNum):
        featList = [example[i] for example in dataSet]  #每个feature的list
        uniqueVals = set(featList)                      #每个list的唯一值集合                 
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  #每个唯一值对应的剩余feature的组成子集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcEntropy(subDataSet)
            splitInfo += -prob * math.log(prob, 2)
        infoGain = baseEntropy - newEntropy              #这个feature的infoGain
        if (splitInfo == 0):                             #说明splitDataSet无法继续拆分(featureNum = 1)
            bestFeature = i                              #暂定当前特征为最佳特征
            continue                                     #跳过本次循环,避免除以0报错
        infoGainRatio = infoGain / splitInfo             #这个feature的infoGainRatio
        if (infoGainRatio >= bestInfoGainRatio):         #选择最大的gain ratio
            bestInfoGainRatio = infoGainRatio
            bestFeature = i                              #选择最大的gain ratio对应的feature
    return bestFeature

#得到List中出现最多的元素
#传入参数：List
def majorityClass(classList):
    return max(map(lambda x: (classList.count(x), x), classList))[1]

#创建树
#传入参数：进行决策树建立的数据List、对应的特征名称List
def createTree(dataSet, featLabels):
    labels = featLabels[:]                                      #临时特征表，避免影响原特征表
    classList = [example[-1] for example in dataSet]            #列出该分支所有特征
    if classList.count(classList[0]) == len(classList):         #classList所有元素(类型)都相等，即类别完全相同，停止划分
        return classList[0]                                     #直接返回类型
    if len(dataSet[0]) == 1:                                    
        return majorityClass(classList)                         #返回出现次数最多的类型
    bestFeat = chooseBestFeatureToSplit(dataSet)             
    bestFeatLabel = labels[bestFeat]                            #选择最大的gain ratio对应的feature
    myTree = {bestFeatLabel:{}}                                 #构建字典
    del(labels[bestFeat])                                       #删除最佳特征，为下一分支准备
    featValues = [example[bestFeat] for example in dataSet]     #得到所有的feature值
    uniqueVals = set(featValues)                                #去重，得到feature值的种类
    for value in uniqueVals:                                    #遍历feature值，每个值创建分支
        subLabels = labels[:]                                   
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#保存树模型
def saveTree(myTree, save_path = 'Model/Tree/1.tree'):
    f = open(save_path,'w+')  
    f.write(str(myTree))  
    f.close()  

#读取树模型
def getTree(save_path = 'Model/Tree/1.tree'):
    f = open(save_path,'r')  
    a = f.read()  
    myTree = eval(a)  
    f.close() 
    return myTree
#找到最优节点
#传入参数：[[feature,class],[feature,class]...]的list
def findNode(valList):
    sort.quickSort(valList)                                     #按值大小快速排序
    flag = valList[0][1]                                        #记录第一项的Class的值
    nodeList = []                                               #候选的节点列表
    bestNode = -1                                               #记录最优节点的序号，初始化为负
    bestNodeInfo = -1                                           #记录最优节点的infoGain值，初始化为负
    index = 0                                                   #记录节点序号
    count = len(valList)
    for each in valList:                                        #each格式为[featureValue,classValue]
        if each[1] != flag:
            flag = each[1]
            nodeList.append(index)
        index += 1
    if nodeList == None:                                         #不存在节点返回All
        return 'All'
    for node in nodeList:                                       #遍历节点，求每个节点的infoGain
        leftList = valList[:node]
        rightList = valList[node:]
        infoGain = calcEntropy(leftList)*node/count +calcEntropy(rightList)*(count-node)/count
                                                                #节点的infoGain值
        #print(str([valList[node-1][0],valList[node][0]])+":"+str(infoGain))
        if infoGain > bestNodeInfo:
            bestNodeInfo = infoGain
            bestNode = node
    return [valList[bestNode-1][0],valList[bestNode][0]]
            
    
#数据预处理(离散化、缺失数据问题等)
#传入参数：进行预处理的数据List、标识连续数据列序号的List
def dataSetPreprocess(dataSet,contLabels):
    for rowNum in contLabels:
        valList = []        #存放连续数据
        for each in dataSet:                                    
            valList.append([each[rowNum],each[-1]])             #连续特征列表
        node = findNode(valList)                                #找最佳分割节点并返回
        for each in dataSet:                                    #遍历List，根据节点进行离散化
            if node == 'All':
                each[rowNum] = 'All'
            elif each[rowNum] <= node[0]:                         #node左侧值
                each[rowNum] = '<='+str(node[0])
            elif each[rowNum] >= node[1]:                    #node右侧值
                each[rowNum] = '>'+str(node[0])

#通过叶子节点数据判断当前能否被悲观剪枝算法剪枝，以及若剪枝，class值为多少
#传入classify返回的resultList即可
#返回：能剪枝：1,'classVal' or 不能剪枝：0,'0'
def ifCanPep(resultList):
    #totalCount = 0                                                  #样本总数
    LeafNum = len(resultList)                                       #叶子节点数
    leafRightNum = 0
    leafWrongNum = 0                                                 #叶子节点错误数
    penalFactor = 0.5                                               #惩罚因子
    for each in resultList:
        leafRightNum += each[0]                                 
        leafWrongNum += each[1]                                 
    dataNum = leafRightNum + leafWrongNum                           #该节点样本总数
    leafError = (leafWrongNum + LeafNum * penalFactor)                      #叶子节点误判数，考虑惩罚因子
    leafErrorRate = leafError / dataNum                             #叶子节点误判率，即未剪枝误判率
    var = math.sqrt(leafErrorRate * (1 - leafErrorRate) * dataNum)  #标准差
    classValList = []                                               #存放class值的种类
    for each in resultList:
        if each[-1] not in classValList:
            classValList.append(each[-1])
    for val in classValList:
        subRightNum = 0
        subWrongNum = 0                                                #记录相对于当前class值，该节点的总正确错误数
        for each in resultList:
            if each[-1] == val:                                     #遍历到class值相等，总正确数、错误数对应增加
                subRightNum += each[0]                                 #each[0]为正确数
                subWrongNum += each[1]                                 #each[1]为错误数，见resultList结构
            else:
                subRightNum += each[1]
                subWrongNum += each[0]
        subError = subWrongNum + penalFactor
        subErrorRate = subError / dataNum
        #print(leafError)
        #print(var)
        #print(subError)
        #print('\n')
        if (leafError + var > subError):                             #叶子节点误判大于剪枝后子树节点误判
            return 1,val                                          #能剪枝则返回  1,代替节点class的值
    return 0,'0'                                                  #不能剪枝则返回0,'0'

#判断叶子节点的正确数、错误数和class值
#传入参数：决策树、数据、特征表、接收结果的list(接收格式为[[正确数,错误数,分类结果]])
def leafClassify(tree,dataSet,featLabels,resultList):
    featName = list(tree.keys())[0]                                 #特征名
    featValDict = tree[featName]                                    #偶数层dictionary
    featIndex = featLabels.index(featName)                          #当前特征在列表的序号
    for key in list(featValDict.keys()):
        subDataSet = splitDataSet(dataSet,featIndex,key)            #按当前特征值划分数据
        if type(featValDict[key]).__name__ != 'dict':               #遍历到字典value不是字典，则为特征
            rightNum = 0                                            #该叶子节点的正确、错误样本的计数
            wrongNum = 0
            for each in subDataSet:                                    #遍历测试数据
                if each[-1] == featValDict[key]:                    #if测试数据class分类符合决策树判断
                    rightNum += 1
                else:
                    wrongNum += 1
            resultList.append([rightNum,wrongNum,featValDict[key]])  #resultList将记录叶子节点正确错误数
        else:
            tmpFeatLabels = featLabels[:]                           #临时特征表，用于递归调用
            tmpFeatLabels.remove(featName)                                      #剔除当前特征名用于递归调用
            leafClassify(featValDict[key],subDataSet,tmpFeatLabels,resultList)     #递归调用直到class
            
#PEP悲观剪枝
#传入参数：决策树、数据、特征表、接收剪枝列表的list
def Pep(tree,dataSet,featLabels):
    featName = list(tree.keys())[0]                                 #特征名
    featValDict = tree[featName]                                    #偶数层dictionary
    featIndex = featLabels.index(featName)                          #当前特征在列表的序号
    for key in list(featValDict.keys()):
        subDataSet = splitDataSet(dataSet,featIndex,key)            #按当前特征值划分数据
        if type(featValDict[key]).__name__ != 'dict':               #遍历到字典value不是字典，则为特征
            continue
        else:                                                       #遍历到字典value是字典，则为子树
            tmpFeatLabels = featLabels[:]                           #临时特征表，用于递归调用
            tmpFeatLabels.remove(featName)                                      #剔除当前特征名用于递归调用
            resultList = []                                                     #存放当前子树的叶子节点误判数等结果
            leafClassify(featValDict[key],subDataSet,tmpFeatLabels,resultList)     #递归调用直到class
            canPep,classVal = ifCanPep(resultList)                              #判断能否剪枝，接收标识和若剪枝节点的class值
            #print(featValDict[key])
            #print(canPep)                                               
            if canPep == 0:
                Pep(featValDict[key],subDataSet,tmpFeatLabels)                  #当前结点不能剪枝则继续遍历
            else:
                featValDict[key] = classVal

                
#单列数据判断其分类
#传入参数：决策树、单列数据、特征表
#返回class分类结果
def SingleDataClassify(tree,dataList,featLabels):
    featName = list(tree.keys())[0]                                 #特征名
    featValDict = tree[featName]                                    #偶数层dictionary
    featIndex = featLabels.index(featName)                          #当前特征在列表的序号
    for key in list(featValDict.keys()):
        if dataList[featIndex] == key:                              #当前特征值等于数据对应特征值
            if type(featValDict[key]).__name__ != 'dict':
                return featValDict[key]
            subDataList = dataList[:]
            subDataList.pop(featIndex)                              #此处按索引删除（千万不能使用remove，若索引元素在前面有重复会删除重复的第一个！）
            tmpFeatLabels = featLabels[:]                           #临时特征表，用于递归调用
            tmpFeatLabels.remove(featName)                                      #剔除当前特征名用于递归调用
            # print('featValDict[key]:'+str(featValDict[key]))
            # print('subDataList:'+str(subDataList))
            # print('tmpFeatLabels:'+str(tmpFeatLabels)+'\n')
            return SingleDataClassify(featValDict[key],subDataList,tmpFeatLabels)      #递归分类直到某分支为class
            
def DataSetGetResult(tree,dataSet,featLabels):
    right_num = 0
    wrong_num = 0
    None_num = 0
    for row in dataSet:
        result = SingleDataClassify(tree,row[:-1],featLabels)
        print(str(row[:-1])+'actual_results:'+str(row[-1])+'  forecast_result:'+str(result))
        if result == None:
            None_num += 1
        elif row[-1] == result:
            right_num += 1
        else:
            wrong_num += 1
    accuracy = right_num/(right_num+wrong_num)
    print('right:'+str(right_num)+'/'+str(right_num+wrong_num+None_num)+'  wrong:'+str(wrong_num)+'/'+str(right_num+wrong_num+None_num)+'  None:'+str(None_num)+'/'+str(right_num+wrong_num+None_num)+'\n')
    print('Accuracy:'+str(accuracy*100)+'%')
    return accuracy
        
        