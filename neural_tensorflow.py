#coding:utf-8
import tensorflow as tf
import numpy as np

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

def create_matmul(input, in_n, out_n, activate=None):
    w = tf.Variable(tf.random_normal([in_n, out_n]))
    b = tf.Variable(tf.zeros([1, out_n]) + 0.1)
    result = tf.matmul(input, w) + b
    if activate == None:
        return result
    else:
        return activate(result)

class NeuralNetwork:
    def __init__(self, session):
        self.session = session      #初始化传入session  ex: NN = xxx.NeuralNetwork(tf.Session()) 
        self.input_n = 0            #输入节点数
        self.hidden_n = []          #隐层节点数列表
        self.output_n = 0           #输出节点数
        self.input_layer = None     #输入层占位符（相当于前向运算的输入）     input->hidden
        self.hidden_layers = []     #隐层前向运算列表,三层结构时为空          hidden->hidden
        self.output_layer = None    #输出前向运算                             hidden->output
        self.label_layer = None     #实际输出标签占位符（测试数据正确结果的输入）
        self.loss = None            #作为训练标准的loss值
        self.train_control = None   #训练控制
    
    #神经网络模型初始配置
    def setup(self, layers):
        if len(layers) < 3:         #不运行3层以下网络（最基础的3层为input、hidden、output）
            return 'layers could not < 3'
        self.input_n = layers[0]
        self.hidden_n = layers[1:-1]  
        self.output_n = layers[-1]

        self.input_layer = tf.placeholder(tf.float32, [None, self.input_n])                         #输入层占位符（相当于前向运算的输入）
        self.label_layer = tf.placeholder(tf.float32, [None, self.output_n])                        #实际输出标签占位符（测试数据正确结果的输入）

        in_n = self.input_n                                                                         #input到hidden[0]的input节点数，用于构建weight张量
        out_n = self.hidden_n[0]                                                                    #input到hidden[0]的output节点数，用于构建weight张量
        self.hidden_layers.append(create_matmul(self.input_layer, in_n, out_n, activate=tf.nn.softmax))#将构建好的前向运算添加到hidden_layers列表
        for i in range(len(self.hidden_n)-1):                       #如果hidden层不止一层则添加hidden到hidden的前向运算
            in_n = out_n
            out_n = self.hidden_n[i+1]
            inputs = self.hidden_layers[-1]
            self.hidden_layers.append(create_matmul(inputs, in_n, out_n, activate=tf.nn.softmax))

        self.output_layer = create_matmul(self.hidden_layers[-1], self.hidden_n[-1], self.output_n) #将hidden最末层到output层的前向运算写入output_layer
    
    #训练函数
    def train(self, input_data, label_data, limit=1000, learn=0.05):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.label_layer - self.output_layer)), reduction_indices=[1])) #根据label值和output值计算loss
        self.train_control = tf.train.GradientDescentOptimizer(learn).minimize(self.loss)                                   #SGD算法
        #self.train_control = tf.train.AdadeltaOptimizer(learning_rate=learn, rho=0.95, epsilon=1e-08, use_locking=False).minimize(self.loss)                                   #Adadelta算法

        self.session.run(tf.global_variables_initializer())
        for i in range(limit):
            self.session.run(self.train_control, feed_dict={self.input_layer: input_data, self.label_layer: label_data})    #循环学习直到limit次数

    #预测函数
    def predict(self, input):
        return self.session.run(self.output_layer, feed_dict={self.input_layer: input})

    #保存模型
    def save_model(self, save_path = 'Model/NN/1.ckpt'):
        tf.train.Saver().save(self.session, save_path)
    
    #读取模型
    def restore_model(self, save_path = 'Model/NN/1.ckpt'):
        tf.train.Saver().restore(self.session, save_path)
        
    #乱序训练   split指学习组测试组分界点
    def train_random(self, data, split, limit=2000, learn=0.02):
        np.random.shuffle(data)                 #数据乱序
        #学习组
        label_data = data[:split,[len(data[0])-1]]
        #测试组
        label_data_test = data[split:,[len(data[0])-1]]
        
        print('label_data:'+str(label_data))
        print('label_data_test:'+str(label_data_test))
        
        data = np.delete(data,[len(data[0])-1],axis = 1)
        #学习组
        input_data = data[:split]
        #测试组
        input_data_test = data[split:]
        
        print('input_data:'+str(input_data))
        print('input_data_test:'+str(input_data_test))
        
        #调用训练
        self.train(input_data, label_data, limit, learn)
        #调用预测
        index = 0
        right = 0
        wrong = 0
        for row in input_data_test:
            test = np.array([row])
            result = self.predict(test)
            print('actual:'+str(label_data_test[index])+'  predict:'+str(result)+' :'+str(label_data_test[index][0] - result[0][0]))
            if (label_data_test[index][0] - result[0][0] < 0.5) and (label_data_test[index][0] - result[0][0] > -0.5):
                right += 1
                print('right')
            else:
                wrong += 1
                print('wrong')
            index += 1
        print('right:'+str(right)+'wrong:'+str(wrong)+'\nAccuracy:'+str(right/(right+wrong)*100)+'%')
        
#异或测试用
if __name__ == '__main__':
    NN = NeuralNetwork(tf.Session())
    NN.setup([2, 5, 5, 1])
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_data = np.array([[0, 1, 1, 0]]).transpose()
    NN.train(x_data, y_data,limit=2000, learn=0.02)
    test_data = np.array([[0, 1]])
    test_data1 = np.array([[1, 0]])
    test_data2 = np.array([[1, 1]])
    test_data3 = np.array([[0, 0]])
    print(str(test_data)+':'+str(NN.predict(test_data)))
    print(str(test_data1)+':'+str(NN.predict(test_data1)))
    print(str(test_data2)+':'+str(NN.predict(test_data2)))
    print(str(test_data3)+':'+str(NN.predict(test_data3)))
    