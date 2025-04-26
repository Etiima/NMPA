#初始化
import numpy as np
import pandas as pd
import math
import random
import copy
import matplotlib.pyplot as plt
import os
import time
import shutil
import xlwt
def jisuan1(A,A2,D2):  #A2是原始的数据，A是改变后的数据
    A1=copy.copy(A)
    A3=copy.copy(A2)
    D = 0
    time1=0
    timeU=np.zeros(len(A))

    # 计算路径距离和时间
    for i in range(len(A)-1):
        D=D+Distance[A[i]][A[i+1]]
        time1 += time123[A[i]][A[i + 1]]
        time1 += fuwushijian[A[i + 1]]
        timeU[i] = time1

    # 回到起点
    D=D+Distance[A[-1]][A[0]]
    timeU[len(A)-1] = time1+time123[A[-1]][A[0]]

    # 输出最终累计时间
    #print(timeU[-1])

    # 验证时间约束
    for j in range(len(timeU)):
        if timeU[j] > zhouqi[A[j]][1] or timeU[j] < zhouqi[A[j]][0]:
            return A3.copy(), D2
    return A1.copy(), D
def jisuan(A):
    total_distance = 0
    for i in range(len(A) - 1):
        total_distance += Distance[A[i]][A[i + 1]]
    total_distance += Distance[A[-1]][A[0]]  # 返回起点的距离
    return total_distance
def duqu(t): #函数 duqu(t) 的作用是读取指定路径 t 下的文件，并提取其中的城市坐标信息，返回一个包含所有城市坐标的列表。
    # 使用 pandas 读取文件，文件以空格分隔，跳过前 6 行，没有表头
    df = pd.read_csv(t, sep="\s+", skiprows=6, header=None)

    # 提取文件中的第 2 列（索引 1）和第 3 列（索引 2）作为 X 和 Y 坐标
    city_x = np.array(df[1][0:len(df) - 1])  # 获取所有城市的 X 坐标
    city_y = np.array(df[2][0:len(df) - 1])  # 获取所有城市的 Y 坐标

    # 将 X 和 Y 坐标打包成 (x, y) 的形式，并存入列表
    city_location = list(zip(city_x, city_y))  # 生成城市坐标列表
    data = city_location  # 将城市坐标列表赋值给变量 data

    return data  # 返回城市坐标列表
def roulette(select_list):  # 通过概率分配来模拟随机选择，偏向于概率大的个体
    s1 = []  # 初始化一个空列表用于存储逆适应值
    for i in range(len(select_list)):
        s1.append(1 / select_list[i])  # 计算每个个体的逆适应值，并添加到 s1 中

    sum_val = sum(s1)  # 计算所有逆适应值的总和
    random_val = random.random()  # 生成一个 [0, 1) 之间的随机数
    probability = 0  # 初始化累计概率为 0

    # 开始轮盘赌选择
    for i in range(len(select_list)):
        probability += s1[i] / sum_val  # 累加当前个体的选中概率
        if probability >= random_val:  # 如果累计概率大于等于随机数
            return i  # 返回被选中的个体下标
        else:
            continue  # 否则继续检查下一个个体
def Cross(cur, best):  # 从两个父个体生成两个新的子个体,输入当前个体 cur 和最优个体 best
    one = copy.copy(cur)  # 拷贝当前个体，避免修改原始数据
    one1 = copy.copy(best)  # 拷贝最优个体

    # 随机选择两个交叉点
    l = [t for t in range(len(one))]
    t = np.random.choice(l, 2, replace=False)  # 随机选取两个点
    x, y = min(t), max(t)

    # 提取交叉片段
    cross_part = one[x:y]  # 从当前个体中提取片段
    cross_part1 = one1[x:y]  # 从最优个体中提取片段

    # 保留不在交叉片段中的部分，并移除冲突元素
    tmp = [t for t in one if t not in cross_part1]  # 当前个体中非冲突部分
    tmp1 = [t1 for t1 in one1 if t1 not in cross_part]  # 最优个体中非冲突部分

    # 将交叉片段插入到末尾，生成新个体
    tmpx = tmp + cross_part1  # 当前个体非冲突部分加上对方的交叉片段
    tmpx1 = tmp1 + cross_part  # 最优个体非冲突部分加上对方的交叉片段

    return tmpx, tmpx1  # 返回生成的两个新个体
def Variation(s):  # 交叉变异
    c = range(len(s))  # 获取序列的索引范围 [0, len(s)-1]
    index1, index2 = random.sample(c, 2)  # 随机选取两个不同的索引位置

    # 交换两个随机位置的元素
    s[index1], s[index2] = s[index2], s[index1]

    return s  # 返回变异后的序列
def huatu(t):
    shu = []
    for ix in t:
        shu.append(data[ix])
    t1, t2 = list(zip(*shu))
    tz = []
    tz1 = []
    for i1 in range(len(shu) - 1):
        tx = []
        tx1 = []
        tx.append(shu[i1][0])
        tx.append(shu[i1 + 1][0])
        tx1.append(shu[i1][1])
        tx1.append(shu[i1 + 1][1])
        tz.append(tx)
        tz1.append(tx1)
    tx = []
    tx1 = []
    tx.append(shu[-1][0])
    tx.append(shu[0][0])
    tx1.append(shu[-1][1])
    tx1.append(shu[0][1])
    tz.append(tx)
    tz1.append(tx1)
    for index in range(len(shu)):
        plt.scatter(t1[index], t2[index], color='blue', linewidth=0.00001)
        plt.plot(tz[index], tz1[index], color='blue')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('输出2.svg',format='svg',pad_inches=0.0,dpi=150)#输出
    plt.show()
# def chang1(t1,n):
#     tz = {} # 存储路径和对应的权值
#     l = [x for x in range(len(t1))]  # 初始化剩余节点集合
#
#     for j in l:  # 遍历每个起点
#         current = j  # 当前节点为选定的起点
#         result = [current]  # 初始化路径
#         rest = [x for x in range(len(t1))]  # 重置剩余节点集合
#         rest.remove(current)  # 从剩余节点中移除当前起点
#
#         while len(rest) != 0:  # 当还有未访问的节点时
#             tmp_min = math.inf  # 初始化最小距离为无穷大
#             tmp_choose = -1  # 初始化选择的节点
#             for i in rest:  # 遍历剩余节点
#                 if Distance[current][i] < tmp_min:  # 找到距离最近的节点
#                     tmp_min = Distance[current][i]  # 更新最小距离
#                     tmp_choose = i  # 记录最近的节点
#             current = tmp_choose  # 更新当前节点为最近节点
#             result.append(tmp_choose)  # 将最近节点加入路径
#             rest.remove(tmp_choose)  # 从剩余节点集合中移除该节点
#
#         # 忽略无效路径
#         tx,D=jisuan1(A=result, D2=-1, A2=result)
#         if D==-1:  # 忽略无效路径
#             continue
#         tz[D]=tx
#     tzz=[]
#     txx=[]
#     for key in tz.keys():
#         tzz.append(key)
#     tzz.sort()
#     tzz=tzz[0:n]
#     for tz1 in tzz:
#         txx.append(tz[tz1])
#     return txx,tzz
def chang1(t1, n):  # 单次一个起点出发，测试多个不同起点，生成若干条路径并计算其路径总长度，返回所有路径及其对应的总长度。
    ts = []  # 用于存储生成的路径
    tp = []  # 用于存储对应路径的总距离

    rest = [x for x in range(len(t1))]  # 初始化剩余节点集合
    l = random.sample(rest, n)  # 随机选取 n 个起点

    for j in l:  # 遍历每个起点
        current = j  # 当前节点为选定的起点
        result = [current]  # 初始化路径
        rest = [x for x in range(len(t1))]  # 重置剩余节点集合
        rest.remove(current)  # 从剩余节点中移除当前起点

        while len(rest) != 0:  # 当还有未访问的节点时
            tmp_min = math.inf  # 初始化最小距离为无穷大
            tmp_choose = -1  # 初始化选择的节点
            for i in rest:  # 遍历剩余节点
                if Distance[current][i] < tmp_min:  # 找到距离最近的节点
                    tmp_min = Distance[current][i]  # 更新最小距离
                    tmp_choose = i  # 记录最近的节点
            current = tmp_choose  # 更新当前节点为最近节点
            result.append(tmp_choose)  # 将最近节点加入路径
            rest.remove(tmp_choose)  # 从剩余节点集合中移除该节点

        ts.append(result)  # 保存生成的路径
        tz = jisuan(A=result)  # 调用 jisuan 计算路径总距离
        tp.append(tz)  # 保存路径的总距离

    return ts, tp  # 返回生成的路径和对应的总距离
def Rl(path):  # 局部反转操作 输入路径 path
    # 随机选择两个不同的索引，确保反转的区间长度大于 1
    while True:
        start, end = random.sample(range(len(path)), 2)  # 随机选择两个不同的索引
        if abs(start - end) > 1:  # 确保反转的区间长度至少为 2
            break  # 如果满足条件，跳出循环
    # 保证 start 小于 end
    if start > end:
        start, end = end, start  # 交换 start 和 end
    path[start:end + 1] = path[start:end + 1][::-1]  # 反转子区间
    return path
def Rl2(t, tx):  # 输入原始路径 t 和路径总距离 tx
    t1 = copy.copy(t)  # 创建路径的副本
    tj = [tx]  # 存储不同路径的总距离
    tx1 = [t1]  # 存储不同的路径

    ban = list(range(len(t1)))  # 创建路径的索引列表
    cx = random.sample(ban, 2)  # 随机选择两个不同的索引
    cx.sort()  # 确保 cx[0] < cx[1]

    # 分别提取路径的三部分并进行反转
    f1, f2, f3 = t1[:cx[0]], t1[cx[0]:cx[1]], t1[cx[1]:]
    f11, f22, f33 = f1[::-1], f2[::-1], f3[::-1]  # 反转路径部分

    # 定义变异路径的组合方式
    variations = [
        f1 + f22 + f3,
        f1 + f22 + f33,
        f11 + f2 + f3,
        f11 + f22 + f3,
        f11 + f22 + f33,
        f11 + f2 + f33,
        f1 + f2 + f33]

    # 计算每种变异路径的总距离并存储
    for var in variations:
        tj.append(jisuan(A=var))  # 假设 jisuan 计算总距离
        tx1.append(var)

    # 返回路径总距离最小的路径及其距离
    min_distance = min(tj)
    return min_distance, tx1[tj.index(min_distance)]  # 返回最小距离及对应路径
def Rb(tx):   #蚁群信息素更新
    t1=copy.copy(tx)
    ban=[i for i in range(len(t1))]
    # cx=random.sample(ban,random.randint(0,len(ban)-1))#得到的是坐标
    cx = random.sample(ban,random.randint(0,int(0.2*len(data))))
    cx.sort() #复制cx这个要访问的城市，cx是为了迭代过程中的减少
    CX=copy.copy(cx)
    for i in range(len(CX)):
        visit = CX[i] - 1
        protrans = np.zeros(len(cx))
        for k in range(len(cx)):
            # 计算当前城市到剩余城市的（信息素浓度^alpha）*（城市适应度的倒数）^beta
            # etable[visit][unvisit[k]],(alpha+1)是倒数分之一，pheromonetable[visit][unvisit[k]]是从本城市到k城市的信息素
            protrans[k] = np.power(pheromonetable[t1[visit]][t1[cx[k]]], alpha) * np.power(
                etable[t1[visit]][t1[cx[k]]], beta)
        cumsumprobtrans = (protrans / sum(protrans)).cumsum()
        cumsumprobtrans-= np.random.rand()
        # 求出离随机数产生最近的索引值
        next = cx[list(cumsumprobtrans > 0).index(True)]
        t1[CX[i]] = tx[next]
        cx.remove(next)
    return t1
# jieguo1=['pr299','pr439','d198',
#          'a280','lin318','ts225','tsp225','u159']
# diedaici=[1200,1000,600,
#           800,1000,800,800,600]
# pdx=[120,200,350,
#      400,400,400,400,300]
# opx=[48191,107217,15780,
#      2579,42029,126643,3916,42080]
fuwushijian=[0, 8, 4, 19, 14, 2, 1, 17, 18, 8, 14, 1, 1, 13, 13, 17,
             12, 20, 12, 15, 20, 3, 16, 9, 5, 10, 13, 16, 1, 9, 1, 7, 15, 13, 5, 11, 5, 10, 1, 6, 18, 12, 7, 17, 15, 14, 20, 18, 5, 20, 5, 2, 16, 12]
data=[(0,0),(40, 75), (25, 80), (10, 90), (15, 100), (20, 100),
              (30, 90), (50, 90), (50, 95), (55, 100), (60, 100), (75, 90),
              (80, 100), (90, 95), (100, 95), (95, 80), (100, 75), (90, 70),
              (85, 70), (90, 60), (90, 50), (100, 50), (100, 45), (100, 30),
              (95, 30), (100, 10), (95, 10), (90, 10), (85, 10), (80, 10), (80, 35),
              (75, 30), (50, 20), (35, 10), (30, 20), (15, 10), (10, 15), (20, 30),
              (20, 35), (20, 45), (20, 55), (30, 35), (30, 30), (40, 35), (50, 30), (55, 50),
              (60, 60), (70, 55), (70, 60), (70, 75), (60, 80), (50, 75), (40, 60)]
zhouqi=[[0,10000000], [72, 534], [51, 288], [71, 696], [0, 529], [23, 272], [8, 383], [100, 358], [6, 422], [19, 529], [49, 439], [2, 636], [60, 177], [29, 440], [39, 169], [14, 381], [63, 613], [62, 450], [73, 266], [60, 651], [60, 501], [9, 207], [100, 177], [13, 343], [9, 635], [58, 688], [9, 683], [52, 648], [98, 109], [72, 321], [90, 565], [40, 427], [100, 344], [13, 480], [89, 693], [4, 114], [86, 160], [98, 175], [40, 693],
        [71, 678], [61, 432], [100, 410], [82, 633], [82, 313], [47, 227], [6, 274], [86, 577], [7, 648], [69, 360], [71, 219], [78, 128], [65, 141], [77, 596], [5, 246]]
Distance = np.zeros((len(data), len(data)))
time123 = np.zeros((len(data), len(data)))
for i in range(len(data)):
    for j in range(len(data)):
        if i != j:
            Distance[i][j] = math.sqrt((data[i][0] - data[j][0]) ** 2 + (data[i][1] - data[j][1]) ** 2)
        else:
            Distance[i][j] = 100000
for i in range(len(data)):
    for j in range(len(data)):
        if i != j:
            time123[i][j] = math.sqrt((data[i][0] - data[j][0]) ** 2 + (data[i][1] - data[j][1]) ** 2)/10
        else:
            time123[i][j] = 100000
# 蚁群算法的初始化
tx1=[34, 33, 35, 0, 36, 37, 38, 39, 40, 52, 45, 46, 47, 48, 49, 50, 51, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 44, 43, 41, 42]
jieguo1,jieguo2=jisuan1(A=tx1,A2=tx1,
                    D2=61.23)

zhong1 = []  # 存最后结果得
zhongtime = []  # 存最后时间得
zuihaodie = []  # 出现最优迭代地方
zuihaotime = []  # 出现最好的时间

jieguo1111=r'D:\个人文件\02.研究生\06.个人相关论文\03.面向机器人任务分配的新型海洋捕食者算法研究\06.程序\信息素环境对比X'
if os.path.exists(jieguo1111):  # 如果存在，删除目录及其内容
    shutil.rmtree(jieguo1111)
outp = os.path.join(jieguo1111)  # 输出路径
os.makedirs(outp)  # 创建输出目录
for cishu in range(5):
    timex = [0]  # 存时间
    start = time.time()

    etable = 1.0 / Distance
    pheromonetable = np.ones((len(data), len(data)))  # 初始化信息素矩阵

    alpha = 5
    beta = 4
    # 海洋捕食者的初始化
    n = 5  # 种群数量
    P = 10
    FADs = 0.2

    Prey, st_p = chang1(data, n)  # 猎物的位置和可捕猎的舒适度
    jie = st_p.index(min(st_p))

    Elite = [Prey[jie] for i in range(n)]  # 精英捕食者
    st_e = [min(st_p) for i in range(n)]  # 精英捕食者的舒适度

    st_zong = []
    st_zongj = []
    die = 0

    st_j = [min(st_p)]  # 存储最优值
    st_x = [Prey[jie]]  # 存储最优解
    maxiter=200
    while die < maxiter:
        # print(Elite)
        die = die + 1
        st_die_0 = {}  # 当前迭代的舒适度
        st_die_1={}
        for i in range(n):
            s1z = Prey[i].copy()
            s2z = [st_p[i]]
            st_die_0[i]=s2z
            st_die_1[i] = [s1z]
        if die < maxiter / 3:
            for i in range(n):
                # for j in range(int(0.5*random.randint(0,10))):
                for j in range(100):
                    # 准备工作：PR为猎物，PC为捕食者，PRJ为猎物的舒适度，PCJ为捕食者的舒适度
                    PR = copy.copy(Prey[i])
                    EL = copy.copy(Elite[i])
                    PRJ = st_p[i]
                    ELJ = st_e[i]
                    # 挑选精英捕食者：Elite1为变异的捕食者，Elite1J为变异捕食者的舒适度。
                    Elite1x = Variation(EL.copy())
                    Elite1,Elite1J = jisuan1(A=Elite1x.copy(), D2=ELJ,A2=copy.copy(EL))
                    st_die_0[i].append(Elite1J)
                    st_die_1[i].append(Elite1.copy())
                    if Elite1J < ELJ:
                        Elite[i] = Elite1.copy()
                        st_e[i] = Elite1J
                    # # 猎物进化：
                    Prey1X = Rb(PR)
                    Prey1,Prey1J= jisuan1(A=Prey1X.copy(), D2=PRJ,A2=PR.copy())
                    st_die_0[i].append(Prey1J)
                    st_die_1[i].append(Prey1.copy())
                    if Prey1J < PRJ:
                        Prey[i] = Prey1.copy()
                        st_p[i] = Prey1J
                    prey, elite = Cross(Elite1x, Prey1)
                    prey,prey1j = jisuan1(A=prey.copy(), D2=Prey1J,A2=copy.copy(Prey1))
                    st_die_0[i].append(prey1j)
                    st_die_1[i].append(prey.copy())
                    if prey1j < PRJ:
                        Prey[i] = prey.copy()
                        st_p[i] = prey1j
                    # for ty in range(len(Prey)):
                    #     PX=jisuan(Prey[i])
                    #     if PX!=st_p[i]:
                    #         print(PX)
                    # for ty in range(len(Elite)):
                    #     PX = jisuan(Elite[i])
                    #     if PX != st_e[i]:
                    #         print(PX)

                        # 第二阶段
        if maxiter / 3 <= die < 2 * maxiter / 3:
            for i in range(int(n / 2)):
                for j in range(100):
                    # 准备工作：PR为猎物，PC为捕食者，PRJ为猎物的舒适度，PCJ为捕食者的舒适度
                    PR = copy.copy(Prey[i])
                    PC = copy.copy(Elite[i])
                    PRJ = st_p[i]
                    PCJ = st_e[i]
                    Prey1J, Preyx =Rl2(t=PR.copy(), tx=PRJ)
                    st_die_0[i].append(Prey1J)
                    st_die_1[i].append(Preyx.copy())
                    if Prey1J < PRJ:
                        Prey[i] = Preyx.copy()
                        st_p[i] = Prey1J
                    Elite1x = Rl(path=PC.copy())
                    Elite1,Elite1J= jisuan1(A=Elite1x.copy(), D2=PCJ, A2=PC.copy())
                    st_die_0[i].append(Elite1J)
                    st_die_1[i].append(Elite1.copy())
                    if Elite1J < PCJ:
                        Elite[i] = Elite1.copy()
                        st_e[i] = Elite1J
                    preyx, elite = Cross(Preyx.copy(), Elite1.copy())
                    prey,prey1j = jisuan1(A=preyx.copy(), D2=Prey1J, A2=Preyx.copy())
                    st_die_0[i].append(prey1j)
                    st_die_1[i].append(prey.copy())
                    if prey1j < PRJ:
                        Prey[i] = prey.copy()
                        st_p[i] = prey1j
            for i in range(int(n / 2), n):
                for j in range(100):
                    PR = copy.copy(Prey[i])
                    PC = copy.copy(Elite[i])
                    PRJ = st_p[i]
                    PCJ = st_e[i]
                    Elite1X = Rb(tx=PC.copy())
                    Elite1,Elite1J = jisuan1(A=Elite1X.copy(), D2=PCJ, A2=PC.copy())
                    st_die_0[i].append(Elite1J)
                    st_die_1[i].append(Elite1.copy())
                    Preyx1 = Variation(PR.copy())
                    Preyx,Prey1J, = jisuan1(A= Preyx1.copy(), D2=PRJ, A2=PR.copy())
                    st_die_0[i].append(Prey1J)
                    st_die_1[i].append(Preyx.copy())
                    if Prey1J < PRJ:
                        Prey[i] = Preyx.copy()
                        st_p[i] = Prey1J
                    prey, eliteX = Cross(Preyx.copy(), Elite1.copy())
                    elite,elite1j = jisuan1(A=eliteX.copy(), D2=Elite1J, A2=Elite1.copy())
                    st_die_0[i].append(elite1j)
                    st_die_1[i].append(elite.copy())
                    if elite1j < PRJ:
                        Prey[i] = elite.copy()
                        st_p[i] = elite1j
        if die >= 2 * maxiter / 3:
            for i in range(n):
                # for j in range(int(0.5*random.randint(0,10))):
                for j in range(100):
                    PR = copy.copy(Prey[i])
                    PC = copy.copy(Elite[i])
                    PRJ = st_p[i]
                    PCJ = st_e[i]
                    Elite1J, Elite1 = Rl2(t=PC.copy(), tx=PCJ)
                    st_die_0[i].append(Elite1J)
                    st_die_1[i].append(Elite1.copy())
                    if Elite1J < PCJ:
                        Elite[i] = Elite1.copy()
                        st_e[i] = Elite1J
                    Preyx = Rl(path=PR.copy())
                    Preyx,Prey1J = jisuan1(A=Preyx.copy(), D2=PRJ, A2=PR.copy())
                    st_die_0[i].append(Prey1J)
                    st_die_1[i].append(Preyx.copy())
                    if Prey1J < PRJ:
                        Prey[i] = Preyx.copy()
                        st_p[i] = Prey1J
                    prey, elite = Cross(Preyx.copy(), Elite1.copy())
                    jisuan1(A=elite.copy(), D2=Elite1J, A2= Elite1.copy())
                    elite,elite1j =jisuan1(A=elite.copy(), D2=Elite1J, A2= Elite1.copy())
                    st_die_0[i].append(elite1j)
                    st_die_1[i].append(elite.copy())
                    if elite1j < PRJ:
                        Prey[i] = elite.copy()
                        st_p[i] = elite1j
        # print(st_die_0)
        # print(st_die_1)
        # cq=st_die
        # for xxt in range(len(st_die_0)):
        #     for t in range(len(st_die_0[xxt])):
        #         jian1=jisuan(st_die_1[xxt][t])
        #         if jian1!=st_die_0[xxt][t]:
        #             print(die)
        #             print('jieshu1')
        random_num = random.random()
        if random_num <= FADs:
            ban = [i for i in range(n)]
            # cx=random.sample(ban,random.randint(0,len(ban)-1))#得到的是坐标
            cx = random.sample(ban, random.randint(0, n-1))
            for it in cx:
                prey1x, prey2x = Cross(st_die_1[it][st_die_0[it].index(min(st_die_0[it]))], st_die_1[it][st_die_0[it].index(max(st_die_0[it]))])
                prey1,prey1j = jisuan1(A=prey1x, D2=min(st_die_0[it]),A2=st_die_1[it][st_die_0[it].index(min(st_die_0[it]))])
                prey2,prey2j = jisuan1(A=prey2x, D2=max(st_die_0[it]),A2=st_die_1[it][st_die_0[it].index(max(st_die_0[it]))])
                if prey1j < prey2j:
                    prex = prey1j
                    pry = prey1.copy()
                else:
                    prex = prey2j
                    pry = prey2.copy()
                if prex < st_p[it]:
                    Prey[it] = pry
                    st_p[it] = prex
        else:
            jie1 = copy.copy(st_p)
            tx1 = roulette(jie1)
            jie1.pop(tx1)
            tx2 = roulette(jie1)
            jiaohuan=0
            prey1x, prey2x = Cross(copy.copy(Prey[tx1]), copy.copy(Prey[tx2]))
            prey1,prey1j =  jisuan1(A=prey1x.copy(), D2=st_p[tx1],A2=copy.copy(Prey[tx1]))
            prey2,prey2j = jisuan1(A=prey2x.copy(), D2=st_p[tx2], A2=copy.copy(Prey[tx2]))
            Prey[tx1] = copy.copy(prey1)
            st_p[tx1] = prey1j
            Prey[tx2] = copy.copy(prey2)
            st_p[tx2] = prey2j
        end1 = time.time()
        timex.append(round(end1 - start, 2))
        st_j.append(min(st_p))
        st1 = Prey[st_p.index(min(st_p))]
        st_x.append(st1.copy())
        Elite = [Prey[st_p.index(min(st_p))] for i in range(n)]
        st_e = [min(st_p) for i in range(n)]

        if st_j[-2] != st_j[-1]:
            changepheromonetable = np.zeros((len(Prey[-1]), len(Prey[-1])))
            for i in range(len(Prey[-1]) - 1):
                changepheromonetable[st1[i]][st1[i + 1]] += 1 / min(st_p)
            changepheromonetable[st1[0]][st1[-1]] = 1 / min(st_p)
            pheromonetable = (1 - 0.001) * pheromonetable + changepheromonetable
        # 记录程序结束的时间
    end = time.time()

    # 记录最优解出现的迭代次数
    zuihaodie.append(st_j.index(min(st_j)))

    # 记录最优解出现的时间
    zuihaotime.append(timex[st_j.index(min(st_j))])

    # 计算程序运行时间，保留两位小数
    endx = round(end - start, 2)

    # 计算最优解的误差率，opt为实际最优值，st_j为当前的适应度值
    #wucha = (min(st_j) - opt) / opt

    # 记录当前迭代的最优适应度
    zhong1.append(min(st_j))

    # 记录每次迭代所花费的时间
    zhongtime.append(endx)

    # 记录每次迭代的误差率
    #zhongwu.append(wucha)

    # 为输出文件命名：根据当前迭代次数来生成文件名
    t2 = '时间' + str(cishu) + '.txt'  # 存储每次结果的时间
    t3 = '迭代结果' + str(cishu) + '.txt'  # 存储每次迭代结果
    t4 = '寻优结果' + str(cishu) + '.txt'  # 存储最终的寻优结果

    # 拼接文件路径
    out2 = os.path.join(outp, t2)
    out3 = os.path.join(outp, t3)
    out4 = os.path.join(outp, t4)

    # 将每次结果的时间写入文件
    f2 = open(out2, "w")
    x2 = str(timex)
    f2.write(x2)
    f2.close()

    # 将每次迭代结果写入文件
    f3 = open(out3, 'w')
    x3 = str(st_j)
    f3.write(x3)
    f3.close()

    # 将最终寻优结果写入文件
    f4 = open(out4, 'w')
    x4 = str(st_x[-1])
    f4.write(x4)
    f4.close()

# 创建一个Excel工作簿，指定编码为'utf-8'
workbook = xlwt.Workbook(encoding='utf-8')

# 在工作簿中添加一个名为'test_sheet'的工作表
worksheet = workbook.add_sheet("test_sheet")

worksheet.write(0, 0, label='最终结果')
worksheet.write(0, 1, label='总时间')
#worksheet.write(0, 2, label='误差率')
worksheet.write(0, 3, label='最优时间')
worksheet.write(0, 4, label='最优迭代')
#worksheet.write(0, 5, label='最优解')
#worksheet.write(0, 6, label='p的迭代次数')
#worksheet.write(0, 7, label='总迭代')
#worksheet.write(0, 8, label='实际总迭代')
for i in range(len(zhong1)):
    worksheet.write(i + 1, 0, zhong1[i])
    worksheet.write(i + 1, 1, zhongtime[i])
    #worksheet.write(i + 1, 2, zhongwu[i])
    worksheet.write(i + 1, 3, zuihaotime[i])
    worksheet.write(i + 1, 4, zuihaodie[i])
    # worksheet.write(i + 1, 5, opt)
    # worksheet.write(i + 1, 6, pdie)
    # worksheet.write(i + 1, 7, maxiter)
    # worksheet.write(i + 1, 8, 0.8*maxiter)
dizhix1 = os.path.join(outp, '结果.xls')
workbook.save(dizhix1)

plt.figure(figsize=(40, 40), dpi=100)
plt.plot(st_j)
plt.show()

# 将 st_j 和 st_x 打包成对，并按 st_j 排序
sorted_pairs = sorted(zip(st_j, st_x))  # 组合并按 st_j 排序
st_j, st_x = zip(*sorted_pairs)  # 解压排序后的结果
# 转为列表（可选，方便后续操作）
st_j = list(st_j)
st_x = list(st_x)

huatu(st_x[0])
#huatu(st_x[-1])
