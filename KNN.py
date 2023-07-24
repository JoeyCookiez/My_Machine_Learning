import math
import numpy as np

class KNN:
    """1, 计算训练样本和测试样本中每个样本点的距离（常见的距离度量有欧式距离，马氏距离等）；
    2, 对上面所有的距离值进行排序；
    3, 选前k个最小距离的样本；
    4, 根据这k个样本的标签进行投票，得到最后的分类类别；"""
    def __init__(self,train_data,test_data,K):
        self.train_data=train_data
        self.test_data=test_data
        self.K=K

    def cal_Euclidean_distance(self,vector1:np.array,vector2:np.array)->float:
        """计算欧几里得距离 vector=[x1,x2,···,xn] vector1与vector2的维度必须相等"""
        return math.sqrt(sum([(vector1[i]-vector2[i])**2 for i in range(len(vector1))]))

    def cal_Manhattan_distance(self,vector1,vector2)->float:
        """计算曼哈顿距离"""
        return sum([(vector1[i]-vector2[i]) for i in range(len(vector1))])

    def cal_distance_between_train_and_test(self)->list:
        """计算训练集和测试集的距离
           train_data的shape为[([x1,···,xn],type)···]
           test_data的shape同理
        """
        # 默认测试数据为一组向量
        K_min_vector=[] #测试数据集的K_min
        K_data_vector=[]
        for test_vector in self.test_data:
            # K_min=[inf,inf,inf,inf,inf]
            K_data = ['' for i in range(self.K)]
            K_min = [float("inf") for i in range(self.K)]  # 存放K个最小距离的样本
            # 1-round [1,inf,inf,inf,inf] dist=1
            # 2-round [1,2,inf,inf,inf] dist=2
            # 3-round [1,2,1,inf,inf] dist=1
            # 4-round [1,2,1,4,inf] dist=4
            # 5-round [1,2,1,4,5] dist=5
            # 6-round [1,2,1,4,5] dist=6
            # 7-round [0.81,2,1,4,5] dist=0.81
            for i in range(len(self.train_data)):
                dist=self.cal_Euclidean_distance(self.train_data[i][0],test_vector[0])
                # 求与test_data最近的K个train_data
                if dist<max(K_min):
                    K_min[K_min.index(max(K_min))]=dist
                    K_data[K_min.index(max(K_min))]=train_data[i]
                # print(K_min)
            K_min_vector.append(K_min)
            K_data_vector.append(K_data)
        # print('K_data_vector',K_data_vector,len(K_data_vector))
        return K_data_vector

    def predict(self):
        """K_data的shape为[([x1,···,xn],type)···]"""
        # 定义一个字典用来统计各个type的票数
        K_data=self.cal_distance_between_train_and_test()
        predict_arr=[]
        for tr_dat in K_data:
            data_map={}
            for data in tr_dat:
                data_map[data[1]]=0
            for data in tr_dat:
                data_map[data[1]]+=1
            predict_arr.append(max(data_map,key=data_map.get))
        print('test_data',self.test_data)
        print('predict_arr',predict_arr)

def pre_process_dataset(path)->list:
    """读取data文件转成list或者np.arrary类型"""
    dataset=[]
    with open(path,'r')as f:
        data_lines=f.readlines()
        for data_line in data_lines:
            # print(data_line.split(','))
            vector,type_=list(map(eval,data_line.strip().split(',')[:len(data_line.strip().split(','))-1])),data_line.strip().split(',')[len(data_line.strip().split(','))-1]
            dataset.append((vector,type_))
        f.close()
    return dataset

if __name__ == '__main__':
    train_data=pre_process_dataset('iris\\train.data')
    test_data=pre_process_dataset('iris\\test.data')
    # print(train_data)
    # print(test_data)
    # test_arr=[float('inf') for i in range(5)]
    # print(test_arr.index(max(test_arr)))
    K_nn=KNN(train_data=train_data,test_data=test_data,K=6)
    # K_nn.cal_distance_between_train_and_test()
    K_nn.predict()
