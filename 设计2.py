#初始化
import numpy as np
import pandas as pd
import math
import random
import copy
import matplotlib.pyplot as plt
import os
import time
import xlwt
def jisuan(A):
    D = 0
    for i in range(len(A)-1):
        D=D+Distance[A[i]][A[i+1]]
    D=D+Distance[A[-1]][A[0]]
    return D
def duqu(t):#t是数据的位置
    df = pd.read_csv(t, sep="\s+", skiprows=6, header=None)
    city_x = np.array(df[1][0:len(df)-1])
    city_y = np.array(df[2][0:len(df)-1])
    city_location = list(zip(city_x, city_y))
    data = city_location
    return data
def roulette(select_list):
    s1=[]
    for i in range(len(select_list)):
        s1.append(1/select_list[i])
    sum_val = sum(s1)
    random_val = random.random()
    probability = 0#累计概率
    for i in range(len(select_list)):
        probability += s1[i] / sum_val#加上该个体的选中概率
        if probability >= random_val:
            return i#返回被选中的下标
        else:
            continue
def Cross(cur, best):
    one = copy.copy(cur)
    one1=copy.copy(best)
    l = [t for t in range(len(one))]
    t = np.random.choice(l,2)
    x = min(t)
    y = max(t)
    cross_part=best[x:y]
    cross_part1=one1[x:y]
    tmp = []
    tmp1=[]
    for t in one:
        if t in cross_part:
            continue
        tmp.append(t)
    for t1 in best:
        if t1 in cross_part1:
            continue
        tmp1.append(t1)
        # 两种交叉方法
    tmpx=tmp.copy()
    tmpx1=tmp1.copy()
    tmpx[len(tmp):len(tmp)] = cross_part
    tmpx1[len(tmp1):len(tmp1)] = cross_part1
    return tmpx,tmpx1
def Variation(s):
    c = range(len(s))
    index1,index2 = random.sample(c,2)
    temp = s[index1]
    s[index1] = s[index2]
    s[index2] = temp
    return s
def chang1(t1,n):
    ts=[]
    tp=[]
    rest = [x for x in range(len(t1))]
    l=random.sample(rest,n)
    for j in l:
        current=j
        result = [current]
        rest = [x for x in range(len(t1))]
        rest.remove(current)
        while len(rest) != 0:
            tmp_min = math.inf
            tmp_choose = -1
            for i in rest:
                if Distance[current][i] < tmp_min:
                    tmp_min = Distance[current][i]
                    tmp_choose = i
            current = tmp_choose
            result.append(tmp_choose)
            rest.remove(tmp_choose)
        ts.append(result)
        tz=jisuan(A=result)
        tp.append(tz)
    return ts,tp
def Rl(path):
    start = random.randint(0, len(path))
    while True:
        end = random.randint(0, len(path)-1)
        if np.abs(start - end) > 1:
            break
    if start > end:
        path[end: start + 1] = path[end: start + 1][::-1]
        return path
    else:
        path[start: end + 1] = path[start: end + 1][::-1]
        return path
def Rl2(t,tx):
    t1 = copy.copy(t)
    tj=[tx]
    tx1=[t1]
    ban = [i for i in range(0,len(t1))]
    cx = random.sample(ban, 2)
    cx.sort()
    f1=t1[0:cx[0]]
    f11=f1[::-1]
    f2=t1[cx[0]:cx[1]]
    f22=f2[::-1]
    f3=t1[cx[1]:len(t1)]
    f33=f3[::-1]
    # print(type(f1))
    # print(f22)
    # print(f3)
    d1=f1+f22+f3
    tj.append(jisuan(A=d1))
    tx1.append(d1)
    d2=f1+f22+f33
    tj.append(jisuan(A=d2))
    tx1.append(d2)
    d3=f11+f2+f3
    tj.append(jisuan(A=d3))
    tx1.append(d3)
    d4=f11+f22+f3
    tj.append(jisuan(A=d4))
    tx1.append(d4)
    d5=f11+f22+f33
    tj.append(jisuan(A=d5))
    tx1.append(d5)
    d6=f11+f2+f33
    tj.append(jisuan(A=d6))
    tx1.append(d6)
    d7=f1+f2+f33
    tj.append(jisuan(A=d7))
    tx1.append(d7)
    return min(tj),tx1[tj.index(min(tj))]
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
jieguo1=['初始路径']
diedaici=[200]
pdx=[50]
opx=[42029]
jieguo2=r'C:\Users\tanyun\Desktop\毕业论文\毕业材料提交\矩阵关系'
jieguox=r'D:\谭豪\办公\python\TSP数据'
for tzz in range(len(jieguo1)):
    yinyong=jieguo1[tzz]+'.tsp'
    outp = os.path.join(jieguo2,jieguo1[tzz])
    # os.makedirs(outp)
    data = [(0,0),(1, 23), (25, 23), (7, 14), (13, 14), (19, 14), (13, 12), (25, 13), (7, 7), (19, 7), (25, 1)]
    print(data)
    # Distance = np.zeros((len(data), len(data)))
    pdie=diedaici[tzz]
    maxiter = pdx[tzz]  # 最大迭代次数
    # for i in range(len(data)):
    #     for j in range(len(data)):
    #         if i != j:
    #             Distance[i][j] = math.sqrt((data[i][0] - data[j][0]) ** 2 + (data[i][1] - data[j][1]) ** 2)
    #         else:
    #             Distance[i][j] = 100000
    Distance=[[10000000.0, 24.242640687119287, 36.87005768508881, 16.899494936611667, 20.55634918610405, 25.38477631085024, 19.142135623730955, 30.970562748477146, 11.071067811865476, 22.48528137423857, 26.242640687119284],
          [24.242640687119287, 10000000.0, 25.65685424949238, 12.313708498984763, 17.142135623730955, 21.72792206135786, 17.72792206135786, 28.142135623730958, 19.313708498984763, 25.213203435596434, 35.112698372208094],
         [36.87005768508881, 25.65685424949238, 10000000.0, 23.72792206135786, 17.142135623730955, 11.485281374238571, 16.55634918610405, 10.82842712474619, 27.213203435596434, 19.071067811865476, 23.65685424949238],
         [16.899494936611667, 12.313708498984763, 23.72792206135786, 10000000.0, 7.65685424949238, 12.82842712474619, 7.414213562373095, 20.89949493661167, 7.82842712474619, 14.899494936611667, 23.384776310850242],
         [20.55634918610405, 17.142135623730955, 17.142135623730955, 7.65685424949238, 10000000.0, 6.82842712474619, 2.0, 16.313708498984763, 10.071067811865476, 9.485281374238571, 17.970562748477146],
        [25.38477631085024, 21.72792206135786, 11.485281374238571, 12.82842712474619, 6.82842712474619, 10000000.0, 7.65685424949238, 10.899494936611667, 15.485281374238571, 8.414213562373096, 16.899494936611667],
        [19.142135623730955, 17.72792206135786, 16.55634918610405, 7.414213562373095, 2.0, 7.65685424949238, 10000000.0, 15.727922061357859, 9.242640687119284, 8.071067811865476, 16.55634918610405],
              [30.970562748477146, 28.142135623730958, 10.82842712474619, 20.89949493661167, 16.313708498984763, 10.899494936611667,
               15.727922061357859, 10000000.0, 21.071067811865476, 9.071067811865476, 12.0],

              [11.071067811865476, 19.313708498984763, 27.213203435596434, 7.82842712474619, 10.071067811865476, 15.485281374238571, 9.242640687119284, 21.071067811865476, 10000000.0, 12.828427124746192, 20.48528137423857],
       [22.48528137423857, 25.213203435596434, 19.071067811865476, 14.899494936611667, 9.485281374238571, 8.414213562373096, 8.071067811865476, 9.071067811865476, 12.828427124746192, 10000000.0, 8.485281374238571],
       [26.242640687119284, 35.112698372208094, 23.65685424949238, 23.384776310850242, 17.970562748477146, 16.899494936611667, 16.55634918610405, 12.0, 20.48528137423857, 8.485281374238571, 10000000.0]
    ]
    for i in range(len(Distance)):
        print(len(Distance[i]))
    Distance=np.array(Distance)
    # 蚁群算法的初始化
    zhong1=[]  #存最后结果得
    zhongtime=[]   #存最后时间得
    zhongwu=[]   #存最后误差得
    opt=opx[tzz]
    zuihaodie=[]#出现最优迭代地方
    zuihaotime=[]#出现最好的时间
    for cishu in range(5):
        timex = [0]  # 存时间
        start = time.time()
        etable = 1.0 / Distance
        pheromonetable = np.ones((len(data), len(data)))
        alpha =5
        beta = 4
        # 海洋捕食者的初始化
        n = 5  # 种群数量
        P = 10
        FADs = 0.2
        Prey, st_p = chang1(data, n=n)  # 猎物的位置和可捕猎的舒适度
        jie = st_p.index(min(st_p))
        Elite = [Prey[jie] for i in range(n)]  # 精英捕食者
        st_e = [min(st_p) for i in range(n)]  # 精英捕食者的舒适度
        st_zong = []
        st_zongj = []
        die = 0
        st_j = [min(st_p)]  # 存储最优值
        st_x = [Prey[jie]]  # 存储最优解
        while die <maxiter:
            die=die+1
            st_die = {}  # 当前迭代的舒适度
            for i in range(n):
                s1z = Prey[i].copy()
                s2z = [st_p[i]]
                st_die[i] = [s2z, [s1z]]
            if die < maxiter / 3:
                for i in range(n):
                    # for j in range(int(0.5*random.randint(0,10))):
                    for j in range(pdie):
                        # 准备工作：PR为猎物，PC为捕食者，PRJ为猎物的舒适度，PCJ为捕食者的舒适度
                        PR = copy.copy(Prey[i])
                        EL = copy.copy(Elite[i])
                        sj = random.random()
                        PRJ = st_p[i]
                        ELJ = st_e[i]
                        # 挑选精英捕食者：Elite1为变异的捕食者，Elite1J为变异捕食者的舒适度。
                        Elite1 = Variation(EL)
                        Elite1J = jisuan(A=Elite1)
                        st_die[i][0].append(Elite1J)
                        st_die[i][1].append(Elite1.copy())
                        if Elite1J < ELJ:
                            Elite[i] = Elite1.copy()
                            st_e[i] = Elite1J
                        # 猎物进化：
                        Prey1 = Rb(PR)
                        Prey1J = jisuan(A=Prey1)
                        st_die[i][0].append(Prey1J)
                        st_die[i][1].append(Prey1.copy())
                        if Prey1J < PRJ:
                            Prey[i] = Prey1.copy()
                            st_p[i] = Prey1J
                        prey, elite = Cross(Elite1, Prey1)
                        prey1j = jisuan(A=prey)
                        st_die[i][0].append(prey1j)
                        st_die[i][1].append(prey.copy())
                        if prey1j < PRJ:
                            Prey[i] = prey.copy()
                            st_p[i] = prey1j
                            # 第二阶段
            if maxiter / 3 <= die < 2 * maxiter / 3:
                for i in range(int(n / 2)):
                    for j in range(pdie):
                        # 准备工作：PR为猎物，PC为捕食者，PRJ为猎物的舒适度，PCJ为捕食者的舒适度
                        PR = copy.copy(Prey[i])
                        PC = copy.copy(Elite[i])
                        PRJ = st_p[i]
                        PCJ = st_e[i]
                        Prey1J, Preyx = Rl2(t=PR, tx=PRJ)
                        st_die[i][0].append(Prey1J)
                        st_die[i][1].append(Preyx.copy())
                        if Prey1J < PRJ:
                            Prey[i] = Preyx
                            st_p[i] = Prey1J
                        Elite1 = Rl(path=PC)
                        Elite1J = jisuan(A=Elite1)
                        st_die[i][0].append(Elite1J)
                        st_die[i][1].append(Elite1.copy())
                        if Elite1J < PCJ:
                            Elite[i] = Elite1
                            st_e[i] = Elite1J
                        prey, elite = Cross(Preyx, Elite1)
                        prey1j = jisuan(A=prey)
                        st_die[i][0].append(prey1j)
                        st_die[i][1].append(prey.copy())
                        if prey1j < PRJ:
                            Prey[i] = prey
                            st_p[i] = prey1j
                for i in range(int(n / 2), n):
                    for j in range(pdie):
                        PR = copy.copy(Prey[i])
                        PC = copy.copy(Elite[i])
                        PRJ = jisuan(A=PR)
                        PCJ = jisuan(A=PC)
                        Elite1 = Rb(tx=PC)
                        Elite1J = jisuan(Elite1)
                        st_die[i][0].append(Elite1J)
                        st_die[i][1].append(Elite1.copy())
                        if Elite1J < PCJ:
                            Elite[i] = Elite1
                            st_e[i] = Elite1J
                        Preyx = Variation(PR)
                        Prey1J = jisuan(A=Preyx)
                        st_die[i][0].append(Prey1J)
                        st_die[i][1].append(Preyx.copy())
                        if Prey1J < PRJ:
                            Prey[i] = Preyx
                            st_p[i] = Prey1J
                        prey, elite = Cross(Preyx, Elite1)
                        elite1j = jisuan(A=elite)
                        st_die[i][0].append(elite1j)
                        st_die[i][1].append(elite.copy())
                        if elite1j < PRJ:
                            Prey[i] = elite
                            st_p[i] = elite1j
            if die >= 2 * maxiter / 3:
                for i in range(n):
                    # for j in range(int(0.5*random.randint(0,10))):
                    for j in range(pdie):
                        PR = copy.copy(Prey[i])
                        PC = copy.copy(Elite[i])
                        PRJ = jisuan(A=PR)
                        PCJ = jisuan(A=PC)
                        Elite1J, Elite1 = Rl2(t=PC, tx=PCJ)
                        st_die[i][0].append(Elite1J)
                        st_die[i][1].append(Elite1.copy())
                        if Elite1J < PCJ:
                            Elite[i] = Elite1
                            st_e[i] = Elite1J
                        Preyx = Rl(path=PR)
                        Prey1J = jisuan(A=Preyx)
                        st_die[i][0].append(Prey1J)
                        st_die[i][1].append(Preyx.copy())
                        if Prey1J < PRJ:
                            Prey[i] = Preyx
                            st_p[i] = Prey1J
                        prey, elite = Cross(Preyx, Elite1)
                        elite1j = jisuan(A=elite)
                        st_die[i][0].append(elite1j)
                        st_die[i][1].append(elite.copy())
                        if elite1j < PRJ:
                            Prey[i] = elite
                            st_p[i] = elite1j
            random_num = random.random()
            if random_num <= FADs:
                ban = [i for i in range(n)]
                # cx=random.sample(ban,random.randint(0,len(ban)-1))#得到的是坐标
                cx = random.sample(ban, random.randint(0, n - 1))
                for i in cx:
                    prey1, prey2 = Cross(st_die[i][1][st_die[i][0].index(min(st_die[i][0]))], st_die[i][1][st_die[i][0].index(max(st_die[i][0]))])
                    prey1j = jisuan(A=prey1)
                    prey2j = jisuan(A=prey2)
                    if prey1j < prey2j:
                        prex = prey1j
                        pry = prey1
                    else:
                        prex = prey2j
                        pry = prey2
                    if prex < st_p[i]:
                        Prey[i] = pry
                        st_p[i] = prex
            else:
                jie1 = copy.copy(st_p)
                tx1 = roulette(jie1)
                jie1.pop(tx1)
                tx2 = roulette(jie1)
                prey1, prey2 = Cross(Prey[tx1], Prey[tx2])
                prey1j = jisuan(A=prey1)
                prey2j = jisuan(A=prey2)
                Prey[tx1] = prey1
                st_p[tx1] = prey1j
                Prey[tx2] = prey2
                st_p[tx2] = prey2j
            end1=time.time()
            timex.append(round(end1-start,2))
            st_j.append(min(st_p))
            st1 = Prey[st_p.index(min(st_p))]
            st_x.append(st1)
            Elite = [Prey[st_p.index(min(st_p))] for i in range(n)]
            if st_j[-2] != st_j[-1]:
                changepheromonetable = np.zeros((len(data), len(data)))
                for i in range(len(Prey) - 1):
                    changepheromonetable[st1[i]][st1[i + 1]] += 1 / min(st_p)
                changepheromonetable[st1[0]][st1[-1]] = 1 / min(st_p)
                pheromonetable = (1 - 0.001) * pheromonetable + changepheromonetable
        end=time.time()
        # print(min(st_j))
        # plt.plot(st_j)
        # plt.show()
        zuihaodie.append(st_j.index(min(st_j)))
        zuihaotime.append(timex[st_j.index(min(st_j))])
        endx=round(end-start,2)
        wucha=(min(st_j)-opt)/opt
        zhong1.append(min(st_j))
        zhongtime.append(endx)
        zhongwu.append(wucha)
        # t1 = str(cishu) + '.jpg'  #存图形路径
        t2 = '时间'+str(cishu) + '.txt'  #存每次结果时间
        t3 = '迭代结果' + str(cishu) + '.txt'  #存结果迭代
        t4='寻优结果'+str(cishu)+'.txt'
        # out1 = os.path.join(outp, t1)
        out2 = os.path.join(outp, t2)
        out3 = os.path.join(outp, t3)
        out4 = os.path.join(outp, t4)
        # fig = plt.figure(dpi=150,figsize=(50, 50))
        # ax = fig.add_subplot()
        # ax.plot(st_j, color='black', linewidth=30)
        # fig.savefig(out1, format='jpg', dpi=150)
        f2 = open(out2, "w")
        x2 = str(timex)
        f2.write(x2)
        f2.close()
        f3 = open(out3, 'w')
        x3 = str(st_j)
        f3.write(x3)
        f3.close()
        f4 = open(out4, 'w')
        x4= str(st_x[-1])
        f4.write(x4)
        f4.close()
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet("test_sheet")
    worksheet.write(0, 0, label='最终结果')
    worksheet.write(0, 1, label='总时间')
    worksheet.write(0, 2, label='误差率')
    worksheet.write(0, 3, label='最优时间')
    worksheet.write(0, 4, label='最优迭代')
    worksheet.write(0, 5, label='最优解')
    worksheet.write(0, 6, label='p的迭代次数')
    worksheet.write(0, 7, label='总迭代')
    worksheet.write(0, 8, label='实际总迭代')
    for i in range(len(zhong1)):
        worksheet.write(i + 1, 0, zhong1[i])
        worksheet.write(i + 1, 1, zhongtime[i])
        worksheet.write(i + 1, 2, zhongwu[i])
        worksheet.write(i + 1, 3, zuihaotime[i])
        worksheet.write(i + 1, 4, zuihaodie[i])
        worksheet.write(i + 1, 5, opt)
        worksheet.write(i + 1, 6, pdie)
        worksheet.write(i + 1, 7, maxiter)
        worksheet.write(i + 1, 8, 0.8*maxiter)
    dizhix1 = os.path.join(outp, '结果.xls')
    workbook.save(dizhix1)