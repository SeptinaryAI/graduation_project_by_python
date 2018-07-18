import math
import random
import copy

#随机
def rand(x,y):
    return ((y - x) * random.random() + x)
#将list构造为指定维度的矩阵([]):config的len为矩阵长度，每个元素值为对应维度的元素个数
def list_matrix(config):
    mat = []        #生成的矩阵
    temp = [0.0]       #临时列表
    for i in reversed(config):
        mat = []
        mat.append(temp * i)
        temp = mat
    return mat[0]
#sigmoid函数
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#Relu函数
def Relu(x):
    return (x if(x>0) else 0)
    
#BP神经网络模型类
class BPNeuralNetwork:
    def __init__(self):
        self.input_cell = 0         #输入层节点数
        self.input_weight = []    #输入层权重列表          [[]]
        self.hidden_layer = 0       #隐藏层层数（可多层）
        self.hidden_cell = []       #隐藏层每层节点数列表
        self.hidden_weight = [] #隐藏层每层权重列表      [[[]]]
        self.hidden_bias = []     #隐藏层每层偏差列表      [[]]
        self.output_cell = 0        #输出层节点数
        self.output_bias = []       #输出层偏差列表
    #新建模型函数(int,list,int)
    def create_new(self,i_cell,h_cell,o_cell):
        self.input_cell = i_cell
        self.hidden_layer = len(h_cell)     #h_cell列表长度即为hidden层数
        self.hidden_cell = h_cell
        self.output_cell = o_cell
        # print('self.input_cell:'+str(self.input_cell)+"\n")
        # print('self.hidden_cell:'+str(self.hidden_cell)+"\n")
        # print('self.hidden_layer:'+str(self.hidden_layer)+"\n")
        # print('self.output_cell:'+str(self.output_cell)+"\n")
        self.random_weight_bias(-2.0,2.0)
        self.print_para()
    #输出当前模型的参数
    def print_para(self):
        print('self.input_cell:'+str(self.input_cell)+"\n")
        print('self.input_weight:'+str(self.input_weight)+"\n")
        print('self.hidden_cell:'+str(self.hidden_cell)+"\n")
        print('self.hidden_layer:'+str(self.hidden_layer)+"\n")
        print('self.hidden_weight:'+str(self.hidden_weight)+"\n")
        print('self.hidden_bias:'+str(self.hidden_bias)+"\n")
        print('self.output_cell:'+str(self.output_cell)+"\n")
        print('self.output_bias:'+str(self.output_bias)+"\n")
    
    #自定义权重和偏差
    def set_weight_bias(self):
        self.input_weight = list_matrix([self.input_cell,self.hidden_cell[0]])
        self.hidden_weight = list_matrix([self.hidden_layer])
        self.hidden_bias = list_matrix([self.hidden_layer])
        self.output_bias = list_matrix([self.output_cell])
        self.hidden_bias[0] = list_matrix([self.hidden_cell[0]])
        
        self.input_weight = [[0.2,-0.3],[0.4,0.1],[-0.5,0.2]]
        self.hidden_bias = [[-0.4,0.2]]
        self.hidden_weight = [[[-0.3],[-0.2]]]
        self.output_bias = [0.1]
        self.print_para()
    #初始化随机权重和偏差
    def random_weight_bias(self,x,y):
        self.input_weight = list_matrix([self.input_cell,self.hidden_cell[0]])
        self.hidden_weight = list_matrix([self.hidden_layer])
        self.hidden_bias = list_matrix([self.hidden_layer])
        self.output_bias = list_matrix([self.output_cell])
        self.hidden_bias[0] = list_matrix([self.hidden_cell[0]])
        for i in range(self.input_cell):                #input节点数
                for j in range(self.hidden_cell[0]):    #hidden第一层节点数
                    self.input_weight[i][j] = rand(x,y) #权重在x,y间取值
                    self.hidden_bias[0][j] = rand(x,y)
    
        for i in range(self.hidden_layer):          #hidden层数
            if i == (self.hidden_layer-1):
                self.hidden_weight[i] = list_matrix([self.hidden_cell[i],self.output_cell])
                for j in range(self.hidden_cell[i]): #hidden每层节点数
                    for k in range(self.output_cell):
                        self.hidden_weight[i][j][k] = rand(x,y)
                        self.output_bias[k] = rand(x,y)
            else:
                self.hidden_weight[i] = list_matrix([self.hidden_cell[i],self.hidden_cell[i+1]])
                self.hidden_bias[i+1] = list_matrix([self.hidden_cell[i+1]])
                for j in range(self.hidden_cell[i]): #hidden每层节点数
                    for k in range(self.hidden_cell[i+1]):
                        self.hidden_weight[i][j][k] = rand(x,y)
                        self.hidden_bias[i+1][k] = rand(x,y)
    
    #学习更新权重和偏差(self,[],[])
    def update_weight_bias(self,input_list,output_list,learn):
        #print('input:'+str(input_list)+'  output:'+str(output_list))
        #前向计算每层（不包括input）每个节点的计算值
        out_value_list = copy.deepcopy(self.hidden_bias)
        temp_list = copy.deepcopy(self.output_bias)
        out_value_list.append(temp_list)                             #hidden层与output层有计算出的out值，结构正好与(hidden_bias+output_bias)对应
        #print('out_value_list:'+str(out_value_list))
        
        #input到第一层hidden计算
        for i in range(self.input_cell):                
            for j in range(self.hidden_cell[0]):
                out_value_list[0][j] += (input_list[i] * self.input_weight[i][j]) 
                if i == (self.input_cell - 1):                          #计算完成一个out值都进行非线性化
                    out_value_list[0][j] = sigmoid(out_value_list[0][j])
                    
        #print('out_value_list:'+str(out_value_list))
        #hidden层到output层计算
        for i in range(self.hidden_layer):
            if i == (self.hidden_layer - 1):
                for j in range(self.hidden_cell[i]):
                    for k in range(self.output_cell):
                        out_value_list[i+1][k] += (out_value_list[i][k] * self.hidden_weight[i][j][k])
                        if j == (self.hidden_cell[i] - 1):
                            out_value_list[i+1][k] = sigmoid(out_value_list[i+1][k])
            else:
                for j in range(self.hidden_cell[i]):
                    for k in range(self.hidden_cell[i+1]):
                        out_value_list[i+1][k] += (out_value_list[i][j] * self.hidden_weight[i][j][k])
                        if j == (self.hidden_cell[i] - 1):
                            out_value_list[i+1][k] = sigmoid(out_value_list[i+1][k])
        #print('out_value_list:'+str(out_value_list[-1]))
        
        err_value_list = copy.deepcopy(out_value_list)  #错误list结构正好与out_value_list一样
        #print('err_value:'+str(err_value_list[-1]))
        #反向传递计算
        #output到hidden第一次反向传递
        for i in range(self.output_cell):
            err_value_list[-1][i] = 0.0      #
            err_value_list[-1][i] = out_value_list[-1][i]*(1 - out_value_list[-1][i])*(output_list[i] - out_value_list[-1][i])
            # print('error:'+str(output_list[i] - out_value_list[-1][i]))
            # print('self.hidden_bias:'+str(self.hidden_bias))
            # print('self.output_bias:'+str(self.output_bias))
            self.output_bias[i] += (learn * err_value_list[-1][i])
            # print('self.output_bias[i]:'+str(self.output_bias[i]))
        #hidden到input层的传递
        for ii in range(self.hidden_layer):
            i = self.hidden_layer - ii - 1   #反向处理
            #print('i:'+str(i))
            if i == (self.hidden_layer - 1):
                for j in range(self.hidden_cell[i]):
                    err_value_list[i][j] = 0    #
                    for k in range(self.output_cell):
                        err_value_list[i][j] += (err_value_list[i+1][k] * self.hidden_weight[i][j][k] )
                        self.hidden_weight[i][j][k] += (learn * err_value_list[i+1][k] * out_value_list[i][j])
                    err_value_list[i][j] *= ((1 - out_value_list[i][j])*out_value_list[i][j])
                    self.hidden_bias[i][j] += (learn * err_value_list[i][j])
            else:
                for j in range(self.hidden_cell[i]):
                    err_value_list[i][j] = 0    #
                    for k in range(self.hidden_cell[i+1]):
                        err_value_list[i][j] += (err_value_list[i+1][k] * self.hidden_weight[i][j][k] )
                        self.hidden_weight[i][j][k] += (learn * err_value_list[i+1][k] * out_value_list[i][j])
                    err_value_list[i][j] *= ((1 - out_value_list[i][j])*out_value_list[i][j])
                    self.hidden_bias[i][j] += (learn * err_value_list[i][j])
        #print('err_value_list:'+str(err_value_list))
        #self.print_para()
        #计算input层到hidden的权重更新
        for i in range(self.input_cell):
            for j in range(self.hidden_cell[0]):
                self.input_weight[i][j] += (learn * err_value_list[0][j] * input_list[i])
    
    #通过输入得到输出
    def get_result(self,input_dataset,output_dataset):
        index = 0
        for input_list in input_dataset:
            #前向计算每层（不包括input）每个节点的计算值
            out_value_list = copy.deepcopy(self.hidden_bias)
            temp_list = copy.deepcopy(self.output_bias)
            out_value_list.append(temp_list)                             #hidden层与output层有计算出的out值，结构正好与(hidden_bias+output_bias)对应
            #print('out_value_list:'+str(out_value_list))
            
            #input到第一层hidden计算
            for i in range(self.input_cell):                
                for j in range(self.hidden_cell[0]):
                    out_value_list[0][j] += (input_list[i] * self.input_weight[i][j]) 
                    if i == (self.input_cell - 1):                          #计算完成一个out值都进行非线性化
                        out_value_list[0][j] = sigmoid(out_value_list[0][j])
                        
            #print('out_value_list:'+str(out_value_list))
            #hidden层到output层计算
            for i in range(self.hidden_layer):
                if i == (self.hidden_layer - 1):
                    for j in range(self.hidden_cell[i]):
                        for k in range(self.output_cell):
                            out_value_list[i+1][k] += (out_value_list[i][k] * self.hidden_weight[i][j][k])
                            if j == (self.hidden_cell[i] - 1):
                                out_value_list[i+1][k] = sigmoid(out_value_list[i+1][k])
                else:
                    for j in range(self.hidden_cell[i]):
                        for k in range(self.hidden_cell[i+1]):
                            out_value_list[i+1][k] += (out_value_list[i][j] * self.hidden_weight[i][j][k])
                            if j == (self.hidden_cell[i] - 1):
                                out_value_list[i+1][k] = sigmoid(out_value_list[i+1][k])
            print(str(input_list)+' actual:'+str(output_dataset[index])+' forecast:'+str(out_value_list[-1])+"\n")
            print()
            index += 1
    
    #训练函数
    def train(self,input_dataset,output_dataset,learn,limit):
        count = 0
        i = 0
        while 1:
            self.update_weight_bias(input_dataset[i],output_dataset[i],learn)
            #print('train_input:'+str(input_dataset[i]))
            #print('train_output:'+str(output_dataset[i]))
            count += 1
            i += 1
            if count < limit and i == len(input_dataset):
                i = 0
            elif count >= limit:
                break;
#数据预处理(数据集,离散数据,特征列表)
def dataSetPreprocess(dataSet,discLabels,labelsList):
    labelsList_new = []
    for i in range(len(dataSet[0])):
        if i in discLabels:     #完全离散数据
            rowNum = i
            valList = []        #存放离散数据
            for each in dataSet:
                if each[rowNum] not in valList:
                    valList.append(each[rowNum])
            #print('valList:'+str(valList))
            for val in valList:
                labelsList_new.append(labelsList[rowNum]+'::'+val)
            for each in dataSet:
                replace = []    #用来替换离散变量的多个变量list
                for num in range(len(valList)): #添加与value种类数相同的元素
                    replace.append(0)
                #print('len(valList)'+str(len(valList)))
                #print('replace:'+str(replace))
                for j in range(len(valList)):
                    if valList[j] == each[rowNum]:  #匹配到离散变量的值
                        replace[j] = 1              #替换list对应位置置为1
                each[rowNum] = replace
        else:
            labelsList_new.append(labelsList[i])
    labelsList.clear()
    for each in labelsList_new:
        labelsList.append(each)
    #此处不可使用labelsList = labelsList_new，labelsList会被系统看作临时变量从而无法改变真正的labelsList值
    #将离散化数据处理产生的列表变成多个元素（[[1,2],3,[4,5,6]] --> [1,2,3,4,5,6]）
    dataSet_new = []
    for each in dataSet:
        tmp_list = []
        for i in each:
            if type(i).__name__ != 'list':
                tmp_list.append(float(i))
            else:
                for j in i:
                    tmp_list.append(float(j))
        dataSet_new.append(tmp_list)
    dataSet.clear()
    for each in dataSet_new:
        dataSet.append(each)
    #不能用dataSet = dataSet_new
    #print('labelsList_new:'+str(labelsList_new))
    #print('dataSet:'+str(dataSet))

#需要人工预处理的可量化数据如等级，程度描述，真假等
#(数据集,要量化的数据的列号,量化参考字典)
def dataSetQuantize(dataSet,row,quantizeDict):
    for each in dataSet:
        each[row] = quantizeDict[each[row]] #取对应的量化结果

#超过[0,1]范围的连续数据等比缩放限制到[0,1]内
def dataSetLimit(dataSet,row,area):
    min = area[0]
    max = area[-1]
    for each in dataSet:
        each[row] = (float(each[row]) - min)/(max - min)

if __name__ == '__main__':
    BP1 = BPNeuralNetwork()
    BP1.create_new(2,[5,5],1)
    BP1.train([[1,1],[0,1],[1,0],[0,0]],[[0],[1],[1],[0]],0.05,1000000)
    BP1.get_result([[1,1],[1,0],[0,1],[0,0]],[0,1,1,0])
    # BP1.set_weight_bias()
    # BP1.update_weight_bias([1,0,1],[1],0.9)
