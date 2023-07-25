import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# print(train_images[0])


def show_mnist(train_image, train_labels):
    n = 3
    m = 3
    fig = plt.figure()
    for i in range(n):
        for j in range(m):
            plt.subplot(n, m, i * n + j + 1)
            # plt.subplots_adjust(wspace=0.2, hspace=0.8)
            index = i * n + j  # 当前图片的标号
            img_array = train_image[index]
            img = Image.fromarray(img_array)
            plt.title(train_labels[index])
            plt.imshow(img, cmap='Greys')
    plt.show()


# show_mnist(train_images, train_labels)


class Neuron:
    """定义单个神经元类"""

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def sigmod(self, x) -> float:
        """激活函数sigmod"""
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return x if x >= 0 else 0

    def feedforward(self, inputs):
        """正向传播：单个神经元的输出，输入为一个多维向量，输出为一个多维向量"""
        total = np.dot(inputs, self.weights) + self.bias
        return self.sigmod(total)

    def deriv_sigmod(self, inputs):
        """sigmod函数的导数"""
        x = np.dot(inputs, self.weights) + self.bias
        fx = self.sigmod(x)
        return fx * (1 - fx)


class My_BP:
    """weight的大小应该为输入x的大小，这里x假设为[x1,x2,···,xn]，则weights为[w1,w2,···,wn]
       初始化的weight为随机的[w1,···,wn]，bias为随机数
       根据https://zhuanlan.zhihu.com/p/100419971一文知道隐藏层的神经元数量应为输出大小的2/3与输出层的大小的2/3的和
       实际隐藏层的神经元数量应该根据结果进行修正,根据结果是否过拟合而修改神经元的数量
       这里为了减少形参因此将数据与label封装在train_data中，即train_data[i][0]为数据,train_data[i][1]为label
       每个神经元的权重与偏执都不同，假设第i层隐藏层的第j个神经元为Neuron[i][j]，其输入为inputs[i]，权重为Weight[i][j]，Weight[i][j]的shape为weight_size
       hidden_level为隐藏层的层数
    """

    def __init__(self, train_data, test_data, hidden_level, hidden_layer_number, learn_rate=0.1, epochs=1000):
        """train_data:训练集train_data[i][0]为nx1维向量;test_data:测试集;hidden_level:隐藏层的层数
           hidden_layer_number:隐藏层的神经元个数;learn_rate:学习率;epochds:迭代次数
        """
        self.train_data = train_data
        self.test_data = test_data
        self.hidden_level = hidden_level
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.hidden_layer_number = hidden_layer_number
        # 用list存放每层的神经元
        hidden_layer = [[] for i in range(self.hidden_level)]
        # 确定权重向量的大小
        weights_size = len(train_data[0][0])
        # 确定每层隐藏层的深度:神经元的个数
        # print(int(len(train_data[0][0]) * 2 / 3))
        hidden_layer_size = hidden_layer_number
        # 初始化每个隐藏层
        for i in range(self.hidden_level):
            for j in range(hidden_layer_size):
                # 初始化将每层每个神经元的权重为[0.5,0.5,···,0.5]
                if i == 0:
                    inital_weight = [0.5 for i in range(weights_size)]
                    # 隐藏层每层的神经元的权重向量大小并非一样，隐藏层第一层的神经元的权重向量大小与data的大小一致
                    # 之后的每一层的权重大小等于第一层隐藏层的神经元个数
                    hidden_neuron = Neuron(inital_weight, 1)  # 将每个神经元的偏置设置为1
                    hidden_layer[i].append(hidden_neuron)
                else:
                    inital_weight = [0.5 for i in range(hidden_layer_size)]
                    hidden_neuron = Neuron(inital_weight, 1)  # 将每个神经元的偏置设置为1
                    hidden_layer[i].append(hidden_neuron)
        # 初始化输出层,输出层的输入层必定是隐藏层，因此输出层的weights_size与hidden_layer_size相等
        self.y = Neuron([0.5 for i in range(hidden_layer_size)], 1)  # 将输出层的权重初始化为全为0.5的1xn维向量，偏置置为1

        self.hidden_layer = hidden_layer
        # print(hidden_layer)
        weight_size = len(train_data[0][0])
        # weights=random.sample(range(0,1),weight_size)

    def mse_loss(self, list_true, list_predict) -> float:
        """损失函数，即训练的结果与实际的结果的均方差，结果为一组向量"""
        return ((list_true - list_predict) ** 2).mean()

    def counterpropagation(self, neuron):
        """反向传播"""

        pass

    def train(self):
        """训练：epochs为迭代次数"""
        for epoch in range(self.epochs):  # 进行迭代
            for i in range(len(self.train_data)):  # 遍历每个数据
                data = self.train_data[i][0]
                y_true = self.train_data[i][1]
                per_layer_out = []# 每一层的输出，用一个list存放，
                per_layer_deriv_out=[] #每一层的导数输出，同样用list存放
                tmp_layer_out = data #每一层的输出，临时变量
                for layer in range(len(self.hidden_layer)):  # 进入神经网络开始正向传播
                    print(len(self.hidden_layer[layer][0].weights))  # 打印每层隐藏层第一个神经元的weights_size
                    # for neuron in self.hidden_layer[layer]: #遍历每一层隐藏层中的每一个神经元
                    #     tmp=[]
                    #     neuron_output=neuron.feedforward(inputs=data) #计算每个神经元的输出
                    #     tmp.append(neuron_output)
                    #     per_layer_out=tmp
                    tmp_layer_out = [neuron.feedforward(tmp_layer_out) for neuron in
                                     self.hidden_layer[layer]]  # 每层的输出作为下一层的输入
                    tmp_layer_deriv_out=[neuron.deriv_sigmod(tmp_layer_out) for neuron in self.hidden_layer[layer]]  # 每层的导数输出作为下一层的输入
                    per_layer_out.append(tmp_layer_out)
                    per_layer_deriv_out.append(tmp_layer_deriv_out)
                # 输出层y有且有一个神经元，不妨设最后一层神经元输出为[x1,···xj](假设这层隐藏层有j个神经元)
                # 设y的weights=[w1,···,wj]则y_pred=f(x1*w1+···+xj*wj+by)
                # 输出函数是多元函数可记作L(w1,w2,···,wj,b1,b2,···,bj,by)这里只考虑最后一层隐藏层
                # 进行梯度下降就需要求每个权重和偏置的偏导数 这里以w1为例子
                # ∂L/∂w1=(∂L/∂y_pred)*(∂y_pred/∂w1)
                # ∂y_pred/∂w1=x1*f'(x1*w1+···+xj*wj+by)
                # f'(x1*w1+···+xj*wj+by)=y.deriv_sigmod([x1,···,xj])
                # ∂L/∂w1=dL_dy_pred*x1*self.y.deriv_sigmod([x1,···,xj])
                # 偏置的偏导数即 ∂L/∂by=dL_dy_pred*self.y.deriv_sigmod([x1,···,xj])
                y_pred = self.y.feedforward(tmp_layer_out)  # 隐藏层最后一层作为输出层的输入计算出最终结果y_pred
                deriv_y_pred = self.y.deriv_sigmod(tmp_layer_out)  # 预测值的导数，用于反向传播
                dL_dy_pred = -2 * (y_true - y_pred)
                dypred_dy = deriv_y_pred
                # 输出层反向传播
                # 输出层偏置梯度下降
                self.y.bias -= self.learn_rate * dL_dy_pred * dypred_dy
                # 输出层权重梯度下降
                for i,weight in enumerate(self.y.weights):
                    weight-=dL_dy_pred*tmp_layer_out[i]*dypred_dy
                # 进入隐藏层反向传播，从最后一个隐藏层向前传播 layer为每个隐藏层

                for i,layer in enumerate(list(reversed(self.hidden_layer))):
                    # 遍历每个隐藏层的每个神经元
                    for j in range(len(layer)):
                        hidden_neuron=layer[j]
                        # 遍历每个隐藏层的神经元的权重
                        for weight_seri in range(len(hidden_neuron.weights)):
                            dL_dweight=0
                            #        h11        h21         h31
                            #
                            # x1     h12        h22         h32
                            #
                            # x2     h13        h23         h33         y
                            #
                            # x3     h14        h24         h34
                            #
                            #        h15        h25         h35
                            #         2          1           0
                            pass
                            # for i_ in range(i):
                            #     for j_ in list(reversed(self.hidden_layer))[i_]:
                            #         if j_ == weight_seri:



class TreeNode:
    def __init__(self,x,children:list):
        self.val=x
        self.children=children

class NetWork_Tree:
    def __init__(self,NetWork_List:list):
        """Tree_list=[[y],[a1,a2,a3,a4,a5],[b1,b2,b3,b4,b5],[c1,c2,c3,c4,c5],[x1,x2,x3]]"""
        self.Tree_list=NetWork_List
        self.path=[]

    def create_Tree(self):
        # 将所有神经元存放在treenode中
        for i in range(len(self.Tree_list)-1):
            for j in range(len(self.Tree_list[i])):
                self.Tree_list[i][j]=TreeNode(self.Tree_list[i][j],[])
        # 将神经元连接起来
        for i in range(len(self.Tree_list)-1):
            for j in range(len(self.Tree_list[i])):
                self.Tree_list[i][j].children=self.Tree_list[i+1]

    def find_path(self,aim_node:TreeNode,deep=0):
        """找路径函数"""
        pass



def data_processer(data_list: list) -> list:
    """将多维数组打成一维列表"""
    return [np.array(data).flatten() for data in data_list]


if __name__ == '__main__':
    train_data = list(zip(data_processer(train_images), train_labels))
    test_data = list(zip(data_processer(test_images), test_labels))
    Tree_list = [['y'], ['a1', 'a2', 'a3', 'a4', 'a5'], ['b1', 'b2', 'b3', 'b4', 'b5'], ['c1', 'c2', 'c3', 'c4', 'c5'], ['x1', 'x2', 'x3']]
    test_Tree=NetWork_Tree(NetWork_List=Tree_list)
    test_Tree.create_Tree()
    print(test_Tree.Tree_list[0][0].children)
    # print(len(train_data[0][0]),train_data[0][1])
    # my_bp_network=My_BP(train_data=train_data,test_data=test_data,hidden_level=2)
