from functions import *

class Sol:
    def __init__(self, traindata, learning_rate, regularization_factor, hiddenlayersize, cos_st):
        self.datanum = len(traindata)  # 训练集个数
        self.traindata = traindata  # 训练集数据
        self.lrate = learning_rate  # 学习率
        self.cos_st = cos_st  # 是否使用余弦学习率学习策略
        self.startlrate = learning_rate
        self.reg_factor = regularization_factor  # 正则化强度
        self.hiddensize = hiddenlayersize  # 隐藏层神经元个数
        self.batchsize = 100
        self.loss = []
        self.trainacc = []

        # 初始化网络(两层) 0-9 10个手写数字类别
        self.W1 = (np.random.rand(28 * 28, self.hiddensize) - 0.5) * 2/ 28
        self.b1 = np.zeros((1, self.hiddensize))
        self.W2 = (np.random.rand(self.hiddensize, 10) - 0.5) * 2 / np.sqrt(self.hiddensize)
        self.b2 = np.zeros((1, 10))

        self.allW1 = [self.W1]
        self.allW2 = [self.W2]
        self.allb1 = [self.b1]
        self.allb2 = [self.b2]
    
    def train(self, epoch):
        iternum = self.datanum//self.batchsize
        for _ in range(epoch):
            np.random.shuffle(self.traindata)
            for i in range(iternum):
                if self.cos_st:
                    self.lrate = self.startlrate * (1 + math.cos(2 * math.pi * i/(iternum+1)))/2
                # print('epoch ' + str(_+1) + '/'+ str(epoch) + ' iteration ' + str(i+1) + '/'+ str(iternum))
                images = self.traindata[i * self.batchsize: (i+1) * self.batchsize, :-1]
                labels = self.traindata[i * self.batchsize: (i+1) * self.batchsize, -1:]
                self.trainacc.append(self.predict(images, labels))
                self.update(images, labels)
            print('### epoch ' + str(_+1) + ' Done ###')
    
    def update(self, data, labels):
        # data: batchsize*784的nparray 
        # 中间部分激活函数 ReLu
        hiddenlayer_output = np.maximum(np.matmul(data, self.W1) + self.b1, 0)
        outlayer = np.maximum(np.matmul(hiddenlayer_output, self.W2) + self.b2, 0)

        # 最后 Softmax
        scores = np.exp(outlayer)  # self.batchsize * 10
        scores_sum = np.sum(scores, axis=1, keepdims=True)  # self.batchsize * 1
        # classification_prob = scores / scores_sum
        
        # 交叉熵损失函数，data中第i个仅在类别labels[i]取1，其余类为0
        temp = np.empty((self.batchsize, 1))
        for i in range(self.batchsize):
            temp[i] = scores[i][int(labels[i])]/scores_sum[i]
        crossentropy = - np.log(temp)

        # 损失函数 = 交叉熵损失项 + L2正则化项
        loss = np.mean(crossentropy, axis=0)[0] + 0.5 * self.reg_factor * (np.sum( self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        # print('loss is:')
        # print(loss)
        self.loss.append(loss)  # 储存每个batch训练的损失大小

        # 反向传播更新参数

        # 最后一层残差 res 交叉熵损失取softmax激活函数
        res = scores / scores_sum  # self.batchsize * 10
        for i in range(self.batchsize):
            res[i][int(labels[i])] -= 1
        # print(res.shape)
        # print(res)
        res  /= self.batchsize
        
        dW2 = np.matmul( hiddenlayer_output.T, res)  # self.hiddensize * 10
        # print(dW2.shape)
        # print(self.W2.shape)
        db2 = np.sum(res, axis=0, keepdims=True) # 1 * 10 

        # 由递推公式得到第一层的残差
        dh1 = np.dot( res, self.W2.T) # batchsize * self.hiddensize
        dh1[hiddenlayer_output<=0] = 0 # Relu求导的结果

        dW1 = np.dot( data.T, dh1)
        db1 = np.sum( dh1, axis=0, keepdims=True)

        # L2正则化求导项
        dW2 += self.reg_factor * self.W2
        dW1 += self.reg_factor * self.W1

        # 更新参数
        self.W2 += -self.lrate * dW2
        self.W1 += -self.lrate * dW1
        self.b2 += -self.lrate * db2
        self.b1 += -self.lrate * db1

        self.allW1.append(self.W1)
        self.allW2.append(self.W2)
        self.allb1.append(self.b1)
        self.allb2.append(self.b2)
        
        return
    
    def visualization(self):
        plt.plot(self.loss)
        plt.xlabel('batches')
        plt.ylabel('train loss')
        plt.title('learning_rate:' + str(self.startlrate) + ' reg_factor:' + str(self.reg_factor) + ' hiddensize:' + str(self.hiddensize))
        plt.show()

    def predict(self, testdata, testlabel):
        hiddenlayer_output = np.maximum(np.matmul(testdata, self.W1) + self.b1, 0)
        outlayer = np.maximum(np.matmul(hiddenlayer_output, self.W2) + self.b2, 0)
        prediction = np.argmax(outlayer, axis=1).reshape((len(testdata),1))
        accuracy = np.mean(prediction == testlabel)
        return accuracy
    
    def getProcess(self):
        return self.allW1, self.allW2, self.allb1, self.allb2, self.loss, self.trainacc
    
    def savemodel(self):
        paras = {}
        paras['W1'] = self.W1
        paras['W2'] = self.W2
        paras['b1'] = self.b1
        paras['b2'] = self.b2
        with open('bestpara.pkl', 'wb') as f:
            pickle.dump(paras, f)



def findbest(traindata, testdata, testlabel):
    acclist = {}
    for lr in [0.005, 0.01, 0.05]:
        for regul in [0.001, 0.005, 0.01]:
            for hiddensize in [100, 150, 200, 250, 300]:
                a = Sol(traindata, lr, regul, hiddensize, False)
                a.train(5)
                print(str(lr) + '/' + str(regul) + '/' + str(hiddensize) + '模型在测试集准确率为' + str(a.predict(testdata, testlabel)))
                acclist[str(lr) + '/' + str(regul) + '/' + str(hiddensize)] = a.predict(testdata, testlabel)
    return max(list(acclist.values())), list(acclist.keys())[list(acclist.values()).index(max(list(acclist.values())))], acclist

def finalmodel(lr, regu, hsize):
    a = Sol(traindata, lr, regu, hsize, False)
    a.train(5)
    a.savemodel()
    processpara = {}
    processpara['allW1'] = a.getProcess()[0]
    processpara['allW2'] = a.getProcess()[1]
    processpara['allb1'] = a.getProcess()[2]
    processpara['allb2'] = a.getProcess()[3]
    processpara['processloss'] = a.getProcess()[4]
    processpara['trainacc'] = a.getProcess()[5]

    with open('processparameters.pkl', 'wb') as f:
        pickle.dump(processpara, f)
    
    # 可视化最后选取模型训练的loss/acc
    x = processpara['processloss']
    plt.plot(x, label='loss')
    z = processpara['trainacc']
    plt.plot(z, label='acc')
    plt.xlabel('iteration num')
    plt.ylabel('acc/loss ')
    plt.title('training accuary and loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 获取训练集/测试集 nparray型数据和标签
    traindata = images_data('train')
    trainlabel = labels_data('train')
    traindata = np.append(traindata, trainlabel, axis=1)

    testdata = images_data('test')
    testlabel = labels_data('test')


    output = findbest(traindata, testdata, testlabel)  # 网格搜索找到较优的超参 
    acclist = output[2]  # 所有超参组合对应的测试集准确率

    # 可视化参数搜索最佳超参组合 并 存储acclist
    x = acclist.keys()
    y = acclist.values()
    plt.xlabel('hyper. values')
    plt.xticks(rotation=90)
    plt.ylabel('acc')
    plt.plot(x, y)
    plt.show()
    with open('accuarylist.pkl', 'wb') as f:
        pickle.dump(acclist, f)

    # 使用finalmodel函数储存最后选取的模型并可视化过程损失和准确度
    finalmodel(0.05,0.001,250)
    # 在上一步执行之后可以调用predictbymodel对数据集调用训练好模型
    print(predictbymodel(testdata, testlabel))
