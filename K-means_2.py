import random
from math import sqrt

class K_means:
    def __init__(self,K:int,dataset,iter:int):
        """K为聚类数，dataset为数据集，iter为迭代次数"""
        self.K=K
        self.dataset=dataset
        self.verbose=iter

    def cal_distance(self,point1:list,point2:list)->float:
        """计算两点距离"""
        sumOfSquare = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        return sqrt(sumOfSquare)

    def average_points(self,points_arr:list)->tuple:
        """计算均值点"""
        x_ = sum(point[0] for point in points_arr) / len(points_arr)
        y_ = sum(point[1] for point in points_arr) / len(points_arr)
        return (x_, y_)

    def train(self):
        # 随机生成K个中心点的种子
        # seeds=[0,1]
        seeds=random.sample(range(0,len(self.dataset)),self.K)
        # 将随机选择的K个点作为初始中心点
        init_centre = []
        for i, seed in enumerate(seeds):
            init_centre.append(self.dataset[seed])
        print('初始中心点_train', init_centre)
        # 将初始中心点外的点作为'小弟'
        bubs = [self.dataset[i] for i in range(len(self.dataset)) if i not in seeds]
        print('bubs_train', bubs)
        self.iteration(init_centre=init_centre,init_bubs=bubs)



    def iteration(self,init_centre,init_bubs):
        """init_boss=[(),(),()···()]  """
        centre=init_centre
        bubs=init_bubs
        for verb in range(1,self.verbose+1):
            cluster = [[] for i in range(self.K)]
            bubs_distance = []  # 每个小弟到其他中心点的距离,如3个中心点3个小弟则[[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
            # print('centre_%d'%verb,centre)
            # print('分配1_%d' %verb,bubs)
            for bub in bubs:
                bub_to_centre = []  # 单个小弟到其他中心点的距离[dist1,dist2,···,distK]
                for cen in centre:
                    bub_to_centre_dist = self.cal_distance(bub, cen)  # 两点间的距离
                    bub_to_centre.append(bub_to_centre_dist)
                bubs_distance.append(bub_to_centre)
            # print('每个小弟到其他中心点的距离', bubs_distance)

            rearrange = []
            # 将距小弟最近的中心点的小标记作一个列表
            for bub_dist in bubs_distance:
                close_boss = bub_dist.index(min(bub_dist))
                # print(close_boss)
                rearrange.append(close_boss)
            # 更新最新中心点
            new_centre=list(set(self.dataset)-set(bubs))
            # 将最新中心点分配到各个组/簇
            for i, bos in enumerate(new_centre):
                cluster[i].append(bos)
            # 根据列表重新分配组/簇
            for bub, i in zip(bubs, rearrange):
                cluster[i].append(bub)
            # print('cluster1_%d'%verb,cluster)
            # 重新更新小弟集
            tmp_bubs=[]
            # 如果一个簇仅有一个元素那么这个元素就是一个大哥/中心点，有多个元素的簇将取其均值点作为新的大哥/中心点
            # 之前的大哥将被下方至小弟集中
            for clus in cluster:
                # print('clus',clus[0])
                if len(clus) != 1:
                    # print('clus',clus)
                    for cl in clus:
                        tmp_bubs.append(cl)
            bubs=tmp_bubs
            # print('分配2_%d'%verb,bubs)

            print('第%d轮簇'%verb,cluster)
            boss=[]
            # 计算新的中心点放入cluster
            for clus in cluster:
                new_boss = self.average_points(clus)
                # print(new_boss)
                boss.append(new_boss)
            # print('boss',boss)
            centre=boss

if __name__ == '__main__':
    Dataset = [[0, 0], [1, 2], [3, 1], [8, 8], [9, 10], [10, 7]]
    Dataset = [tuple(point) for point in Dataset]
    K_mean=K_means(2,Dataset,3)
    K_mean.train()

# 初始中心点_train [(9, 10), (10, 7)]
# bubs_train [(0, 0), (1, 2), (3, 1), (8, 8)]
# 第1轮簇 [[(9, 10), (8, 8)], [(10, 7), (0, 0), (1, 2), (3, 1)]]
# 第2轮簇 [[(9, 10), (8, 8), (10, 7)], [(0, 0), (1, 2), (3, 1)]]
# 第3轮簇 [[(9, 10), (8, 8), (10, 7)], [(0, 0), (1, 2), (3, 1)]]