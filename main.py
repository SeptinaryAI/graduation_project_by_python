import sqlite3
import tree
import treePlotter
import neural_tensorflow
import sys  
import numpy as np
import tensorflow as tf
import random
import copy

sys.setrecursionlimit(10000000) #递归深度设置

conn = sqlite3.connect('test.db')
print ("Opened database successfully")
c = conn.cursor()

conn.commit()

#sql数据记入List
def dataToList(c,tableName):
    dataSet = []
    cursor = c.execute("select * from "+tableName+" order by ID")
    for row in cursor:
        rowList = []
        for val in row:
            rowList.append(val)
        dataSet.append(rowList)
    return dataSet

dataSet = dataToList(c,'german')
print("Table read successfully")

conn.close()

for each in dataSet:
    del each[0]             #删除数据的ID
print("ID deleted successfully")

#random.shuffle(dataSet)        #数据乱序

#tree.dataSetPreprocess(dataSet,[1,2])
#决策树交叉验证，数据集、占比，例如0.1表示每次取90%数据训练、10%测试
def cross_validation_tree(dataSet,rate,tmpLabels):
    count = int(1/rate)                  #验证次数
    test_len = int(len(dataSet) * rate)  #测试数据集的长度
    train_dataSet = []                  #临时训练数据集
    test_dataSet = []                   #临时测试数据集
    tree_list = []                      #生成树列表
    accuracy_list = []                  #存放准确率
    for i in range(count):
        if i == 0:                      #第一个分割的处理
            train_dataSet = dataSet[test_len:]
            test_dataSet = dataSet[:test_len]
        elif i == count - 1:            #最后分割的处理
            train_dataSet = dataSet[:-test_len]
            test_dataSet = dataSet[-test_len:]
        else:
            train_dataSet = dataSet[:test_len * i]
            train_dataSet.extend(dataSet[test_len*(i+1):])
            test_dataSet = dataSet[test_len * i:test_len * (i+1)]
        
        lastest_tree = tree.createTree(train_dataSet, tmpLabels)
        tree_list.append(lastest_tree)
        #print('lastest_tree:'+str(lastest_tree))
        tree.Pep(lastest_tree,train_dataSet,tmpLabels)
        accuracy_list.append(tree.DataSetGetResult(lastest_tree,test_dataSet,tmpLabels))   #批量得到结果
    #求准确度平均值
    average_accuracy = 0
    max_accuracy = max(accuracy_list)   #最大值
    for each in accuracy_list:
        average_accuracy += each
    average_accuracy /= len(accuracy_list)
    
    print('accuracy:'+str(accuracy_list))
    print('max_accuracy:'+str(max_accuracy*100)+'%')
    print('average_accuracy:'+str(average_accuracy*100)+'%')
    i = 1   #树的临时编号
    for each_tree in tree_list:
        tree.saveTree(each_tree, save_path = 'Model/Tree/cross/'+str(i)+'.tree')
        print("Decision tree list saved successfully！"+'Model/Tree/cross/'+str(i)+'.tree')
        i += 1
    
#神经网络交叉验证，数据集、占比，例如0.1表示每次取90%数据训练、10%测试
def cross_validation_nn(dataSet,rate,tmpLabels):
    count = int(1/rate)                  #验证次数
    test_len = int(len(dataSet) * rate)  #测试数据集的长度
    train_dataSet = []                  #临时训练数据集
    test_dataSet = []                   #临时测试数据集
    accuracy_list = []                  #存放准确率
    for i in range(count):
        if i == 0:                      #第一个分割的处理
            train_dataSet = dataSet[test_len:]
            test_dataSet = dataSet[:test_len]
        elif i == count - 1:            #最后分割的处理
            train_dataSet = dataSet[:-test_len]
            test_dataSet = dataSet[-test_len:]
        else:                           
            train_dataSet = dataSet[:test_len * i]
            train_dataSet.extend(dataSet[test_len*(i+1):])
            test_dataSet = dataSet[test_len * i:test_len * (i+1)]
            
        input_dataSet = []
        output_dataSet = []
        input_dataSet_test = []
        output_dataSet_test = []
        #训练数据
        for each in train_dataSet:
            input_dataSet.append(each[:-1])
            output_dataSet.append([each[-1]])
        
        #测试数据
        for each in test_dataSet:
            input_dataSet_test.append(each[:-1])
            output_dataSet_test.append([each[-1]])
            
        #构建模型
        input_cell = len(input_dataSet[0])  #输入个数最终由预处理后的数据决定
        print('input_cell:'+str(input_cell))
        hidden_cell = 7                    #隐藏层节点数
        output_cell = 1                     #输出层节点数
        tf.set_random_seed(1234)            #随机数种子，保证每次随机值相同，便于对比准确率
        nn = neural_tensorflow.NeuralNetwork(tf.Session())
        nn.setup([input_cell,hidden_cell,output_cell])
        X = np.array(input_dataSet)         #训练数据的特征值
        Y = np.array(output_dataSet)        #训练数据的分类结果标签
        nn.train(X, Y, limit=850, learn=0.05)  #训练
        tmp_accuracy = get_nn_accuracy(nn,input_dataSet_test,output_dataSet_test)   #当前验证准确率
        accuracy_list.append(tmp_accuracy)    #计算准确率并计入列表
        nn.save_model(save_path = 'Model/NN/cross/'+str(i+1)+'.ckpt')                       #保存训练完的模型
        print("model "+str(i+1)+" accuracy:"+str(tmp_accuracy))
        print("NN model saved successfully！"+'Model/NN/cross/'+str(i+1)+'.ckpt')
    
    #求准确度平均值
    average_accuracy = 0
    max_accuracy = max(accuracy_list)   #最大值
    for each in accuracy_list:
        average_accuracy += each
    average_accuracy /= len(accuracy_list)
    
    print('accuracy_list:'+str(accuracy_list))
    print('max_accuracy:'+str(max_accuracy*100)+'%')
    print('average_accuracy:'+str(average_accuracy*100)+'%')

    
#定义两个模型为综合模型
#(决策树字典、神经网络类、综合模型名称)
def set_com_model(desicionTree,nn,name = '1'):
    path = 'Model/Com/' + str(name) + '/'
    if os.path.exists('Model/Com/' + str(name)):
        print('[ERROR]:com_model:'+str(name)+' is already exists !!!')
        return
    tree.saveTree(desicionTree,save_path = path + '0.tree') #保存决策树模型
    nn.save_model(save_path = path + '0.ckpt')              #保存神经网络模型
    print('set com_model:'+str(name)+' successfully!!!')

    
    
    
#结果选择策略
def choose_result(result_tree,result_nn):
    if result_tree == None:
        return result_nn
    #elif result_tree - result_nn < 0.001:
    #    return result_tree
    else:
        return result_tree
        
    
#使用综合模型得到结果
#(综合模型名称，)
def com_model_get_result(dataSet,com_model_name = '1'):
    dataSet_tree = dataSet[:]           #树模型的数据
    dataSet_nn = copy.deepcopy(dataSet)     #神经网络模型的数据（深拷贝）
    #两数据相互对立
    
    labels_tree = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20']          #树模型的特征
    labels_nn = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','result']   #神经网络模型的特征
    
    preprocess_tree(dataSet_tree)           #树模型的数据预处理
    preprocess_nn(dataSet_nn,labels_nn)     #神经网络模型的数据预处理
    
    print('len(dataSet_nn[0])-1:'+str(len(dataSet_nn[0])-1))
    
    desicionTree = tree.getTree('Model/Com/' + str(com_model_name) + '/0.tree')   #恢复决策树模型
    nn = neural_tensorflow.NeuralNetwork(tf.Session())
    nn.setup([len(dataSet_nn[0])-1,8,1])                #定义模型结构和参数，必须与读取的模型匹配
    nn.restore_model(save_path = 'Model/Com/' + str(com_model_name) + '/0.ckpt')  #恢复神经网络模型
    #nn.restore_model(save_path = 'Model/NN/cross/6.ckpt')  #恢复神经网络模型
    
    right = 0
    wrong = 0
    for i in range(len(dataSet)):
        result_tree = tree.SingleDataClassify(desicionTree,dataSet_tree[i],labels_tree) #同一个样本，决策树的分类结果
        result_nn = nn.predict(np.array([dataSet_nn[i][:-1]]))                          #同一个样本，神经网络的分类结果
        result_true = dataSet_nn[i][-1]                                                 #该样本的真实结果
        
        #树模型结构统一化为0和1
        change = {'1':0.0,'2':1.0,None:None}
        result_tree = change[result_tree]
        
        #神经网络模型结构统一化为0和1
        if result_nn < 0.5:
            result_nn = 0.0
        else:
            result_nn = 1.0
        
        result_predict = choose_result(result_tree,result_nn)       #结果选择策略
        
        #print('result_tree:'+str(result_tree))                     #决策树结果
        #print('result_nn:'+str(result_nn))                         #神经网络结果
        print('result_predict:'+str(result_predict))                #通过策略选择的最终结果
        print('result_true:'+str(result_true))                #测试数据的实际结果
        print('\n')
        
        if result_true == result_predict:
            right += 1
        else:
            wrong += 1
    com_accuracy = right/(right+wrong)
    print('com_accuracy:'+str(com_accuracy * 100)+'%')
#决策树数据预处理函数集合
def preprocess_tree(dataSet):
    tree.dataSetPreprocess(dataSet,[1,4,12])

#使用树模型
def tree_model_use():
    #对于树模型，将连续数据离散化(数据集,标记连续数据的list)
    preprocess_tree(dataSet)
    print("dataset Preprocessed successfully")

    labels = ['department','departmentType','zhengzhi','english','class1','class2','firstTest','secTest','total']
    labels2 = ['Outlook','Temperature','Humidity','Windy']
    labels3 = ['loan_amnt','funded_amnt','funded_amnt_rate','term','int_rate','installment','emp_length','home_ownership','annual_inc','verification_status','loan_status','inq_last_6mths','addr_state','dti','delinq_2yrs','open_acc','pub_rec','revol_bal','revol_util','total_acc']
    labels4 = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20']
    tmpLabels = labels4[:]

    cross_validation_tree(dataSet,0.2,tmpLabels)        #调用交叉验证
    #desicionTree = tree.createTree(dataSet[0:799], tmpLabels)
    #print("Decision tree created successfully")

    #tree.Pep(desicionTree,dataSet,tmpLabels)
    #print("Decision tree pruned successfully")

    #tree.saveTree(desicionTree, save_path = 'Model/Tree/3.tree')
    #print("Decision tree saved successfully")
    
    #desicionTree = tree.getTree('Model/Tree/cross/5.tree')
    #print("Decision tree restored successfully")
    
    #classResult = tree.SingleDataClassify(desicionTree,['Overcast',85,85,'F',0],tmpLabels)
    #print(classResult)

    #tree.DataSetGetResult(desicionTree,dataSet[799:999],tmpLabels)   #批量得到结果
    #treePlotter.createPlot(desicionTree)
    


#神经网络判断准确率
#传入(神经网络类，测试数据输入，测试数据分类结果标签)
def get_nn_accuracy(nn,input_dataSet_test,output_dataSet_test):
    index = 0
    right = 0
    wrong = 0
    accuracy = 0
    for row in input_dataSet_test:
        test = np.array([row])
        result = nn.predict(test)
        #print('actual:'+str(output_dataSet_test[index])+'  predict:'+str(result)+' :'+str(output_dataSet_test[index][0] - result[0][0]))
        if (output_dataSet_test[index][0] - result[0][0] < 0.5) and (output_dataSet_test[index][0] - result[0][0] > -0.5):
            right += 1
            #print('right')
        else:
            wrong += 1
            #print('wrong')
        index += 1
    accuracy = right/(right+wrong)
    print('right:'+str(right)+'wrong:'+str(wrong)+'\nAccuracy:'+str((accuracy)*100)+'%')
    return accuracy             #将准确率返回
    
#神经网络数据预处理函数集合
def preprocess_nn(dataSet,tmpLabels):
        #对于离散数据，BP神经网络可将其作为多个输入节点来处理(数据集,标记离散数据的list)
        #neural_tensorflow.dataSetQuantize(dataSet,3,{'T':1,'F':0})
        #neural_tensorflow.dataSetQuantize(dataSet,0,{'A11':-0.5,'A12':0.5,'A13':1,'A14':0})
        #neural_tensorflow.dataSetQuantize(dataSet,5,{'A61':0.05,'A62':0.3,'A63':0.75,'A64':1.0,'A65':0})
        
        neural_tensorflow.dataSetLimit(dataSet,1,[0,60])
        neural_tensorflow.dataSetLimit(dataSet,4,[0,12000])
        neural_tensorflow.dataSetLimit(dataSet,12,[18,76])
        neural_tensorflow.dataSetQuantize(dataSet,20,{'1':0,'2':1.0})
        
        #neural_tensorflow.dataSetLimit(dataSet,1,[9,73])
        #neural_tensorflow.dataSetLimit(dataSet,3,[9,100])
        #neural_tensorflow.dataSetLimit(dataSet,9,[18,76])
        #neural_tensorflow.dataSetQuantize(dataSet,24,{'1':0,'2':1.0})
        
        neural_tensorflow.dataSetPreprocess(dataSet,[0,2,3,5,6,7,8,9,10,11,13,14,15,16,17,18,19],tmpLabels)
        #neural_tensorflow.dataSetPreprocess(dataSet,[0,2,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23],tmpLabels)
        #print('dataSet:'+str(dataSet))
        #print('tmpLabels:'+str(tmpLabels))
        #print('dataSet_new:'+str(dataSet_new))
        
#使用神经网络模型
def neural_model_use():
    labels2 = ['Outlook','Temperature','Humidity','Windy','PlayGolf']   #神经网络预处理包括对结果的处理，相比决策树多一个结果的label
    labels3 = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','result']
    labels4 = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','result']
    tmpLabels = labels4[:]
    
    preprocess_nn(dataSet,tmpLabels)    #数据预处理
    def test_tf():              #正式测试（简单测试，非交叉）
        input_dataSet = []
        output_dataSet = []
        input_dataSet_test = []
        output_dataSet_test = []
        #训练数据
        for each in dataSet[0:799]:
            input_dataSet.append(each[:-1])
            output_dataSet.append([each[-1]])
        
        #测试数据
        for each in dataSet[799:999]:
            input_dataSet_test.append(each[:-1])
            output_dataSet_test.append([each[-1]])
        #print('input_dataSet:'+str(input_dataSet))
        #print('output_dataSet:'+str(output_dataSet))
        print('input_cell:'+str(len(input_dataSet[0])))
        
        input_cell = len(input_dataSet[0])  #输入个数最终由预处理后的数据决定
        output_cell = 1
        
        tf.set_random_seed(1234)
        nn = neural_tensorflow.NeuralNetwork(tf.Session())
        nn.setup([input_cell,14,output_cell])
        #nn.train_random(np.array(dataSet),799,limit = 2000)
        
        #nn.restore_model()      #读取模型
        
        #print('input_cell:'+str(input_cell))
        X = np.array(input_dataSet)
        Y = np.array(output_dataSet)
        # #print('x:'+str(X))
        # #print('y:'+str(Y))
        nn.train(X, Y, limit=1050, learn=0.05)
        nn.save_model()         #保存模型
        
        get_nn_accuracy(nn,input_dataSet_test,output_dataSet_test)
        
    cross_validation_nn(dataSet,0.15,tmpLabels)  #调用交叉验证
    #test_tf()                                  #调用普通验证
    
#neural_model_use()
#tree_model_use()
com_model_get_result(dataSet,com_model_name = '1')