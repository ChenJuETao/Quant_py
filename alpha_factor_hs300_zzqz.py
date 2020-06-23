# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 04:45:36 2020

@author: Lenovo
"""



import numpy as np
import sympy
from sympy import Matrix
from scipy import optimize
import baostock as bs
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import statsmodels.api as sm
import statsmodels.formula.api as smf

#matplotlib中文显示
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
#matplotlib负号显示
plt.rcParams['axes.unicode_minus']=False

plt.style.use('ggplot')

#alpha序列
alpha_series_3 = []
alpha_series_ad = []

#变量个数
p = 3

#投资组合指数或股票
code_se = 'sh.000300'

#频率
freq = 'm'

#读取中证全指数据
zzqz_data = pd.read_csv("D:/量化python/中证全指数据.csv")
zzqz_close = [float(close) for close in zzqz_data['close']]

#读取日期数据
#
#
# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
rs = bs.query_history_k_data_plus("sh.000300",
    "date,code,close",
    start_date='2011-12-01', end_date='2018-01-01', frequency="m")
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
hs300_index = pd.DataFrame(data_list, columns=rs.fields)
date_count = hs300_index['date']
# 登出系统
bs.logout()

#数据长度
n_date = len(date_count)



# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)



for i in range(n_date-12):

    #𝑆𝑀𝐵：代表规模因子，表示小盘股与大盘股收益率之差，分别用中证500指数   
    #和中证 100 指数代替小盘股和大盘股。
    #𝐻𝑀𝐿：代表了市值账面比因子，主要是衡量高市值账面比与低市值账面比的股票收益率之差   
    #中证 800 成长与中证 800 价值指数之差作为替代。
    
    #sh.000905中证500 & sh.000903中证100 
    #sh.000918沪深300成长指数 & sh.000919沪深300价值 数据读取
    


    #中证500
    #
    #
    rs = bs.query_history_k_data_plus("sh.000905",
        "date,code,close",
        start_date=date_count[i], end_date=date_count[i+12], frequency=freq)
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    zz500_index = pd.DataFrame(data_list, columns=rs.fields)
    
    zz500_close = [float(close) for close in zz500_index['close']]
    
    
    
    #中证100
    #
    #
    rs = bs.query_history_k_data_plus("sh.000903",
        "date,code,close",
        start_date=date_count[i], end_date=date_count[i+12], frequency=freq)
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    zz100_index = pd.DataFrame(data_list, columns=rs.fields)
    
    zz100_close = [float(close) for close in zz100_index['close']]
    
    
    
    #沪深300成长
    #
    #
    rs = bs.query_history_k_data_plus("sh.000918",
        "date,code,close",
        start_date=date_count[i], end_date=date_count[i+12], frequency=freq)
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    hs300_gr_index = pd.DataFrame(data_list, columns=rs.fields)
    
    hs300_gr_close = [float(close) for close in hs300_gr_index['close']]
    
    
    
    #沪深300价值
    #
    #
    rs = bs.query_history_k_data_plus("sh.000919",
        "date,code,close",
        start_date=date_count[i], end_date=date_count[i+12], frequency=freq)
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    hs300_vl_index = pd.DataFrame(data_list, columns=rs.fields)
    
    hs300_vl_close = [float(close) for close in hs300_vl_index['close']]

    
    
    #SMB因子计算
    #
    #
    n_zz500 = len(zz500_close)
    R_zz500 = np.log([zz500_close[j]/zz500_close[j-1] for j in range(1, n_zz500)]) 
    n_zz100 = len(zz100_close)
    R_zz100 = np.log([zz100_close[j]/zz100_close[j-1] for j in range(1, n_zz100)]) 
    SMB = R_zz500-R_zz100
    
    #HML因子计算
    n_hs300_vl = len(hs300_vl_close)
    R_hs300_vl = np.log([hs300_vl_close[j]/hs300_vl_close[j-1] for j in range(1, n_hs300_vl)]) 
    n_hs300_gr = len(hs300_gr_close)
    R_hs300_gr = np.log([hs300_gr_close[j]/hs300_gr_close[j-1] for j in range(1, n_hs300_gr)]) 
    HML = R_hs300_vl-R_hs300_gr
    
    #MKT因子计算
    #无风险收益率（十年期国债收益率）
    Rf = 0.0032/n_zz500
    
    # 计算市场基准组合收益率
    Rm = np.log([zzqz_close[j]/zzqz_close[j-1] for j in range(i+5,i+17)]) 
    #print(Rm)
    
    MKT = Rm-np.ones(len(Rm))*Rf
    
    
    
    #投资组合
    #
    #
    rs = bs.query_history_k_data_plus(code_se,
        "date,code,close",
        start_date=date_count[i], end_date=date_count[i+12], frequency=freq)
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    se_index = pd.DataFrame(data_list, columns=rs.fields)
    
    se_close = [float(close) for close in se_index['close']]
    
    
    
    #计算投资组合收益率
    n_se = len(se_close)
    R_se = np.log([se_close[j]/se_close[j-1] for j in range(1, n_se)]) 
    
    #收益率-无风险收益率
    Ri = R_se-np.ones(len(Rm))*Rf
    
    #因子矩阵
    fac_mat = pd.DataFrame({'Ri':Ri,'MKT':MKT,'SMB':SMB,'HML':HML})
    
    
    
    # 计算沪深300 alpha & beta(三因子模型)
    model_se_3 = smf.ols('Ri ~ MKT + SMB + HML', data=fac_mat)
    result_se_3 = model_se_3.fit()  # 拟合
    #result_se_3.summary()
    alpha_series_3.append(result_se_3.params[0])
    
    alpha_1 = result_se_3.params[0]
    MKT_1 = result_se_3.params['MKT']
    SMB_1 = result_se_3.params['SMB']
    HML_1 = result_se_3.params['HML']
    
    #定义符号变量
    e1 = sympy.Symbol('e1')
    e2 = sympy.Symbol('e2')
    e3 = sympy.Symbol('e3')
    m_ad = MKT - np.ones(len(Rm))*e1
    s_ad = SMB - np.ones(len(Rm))*e2
    h_ad = HML - np.ones(len(Rm))*e3
    
    #计算alpha标准差
    N = len(Rm)
    M = Matrix([
    np.ones(len(Rm)),
    m_ad,
    s_ad,
    h_ad])
    M = M.T #因子矩阵
    beta_m = Matrix([
    alpha_1,
    MKT_1,
    SMB_1,
    HML_1]) 
    beta_m =beta_m.T #暴露度矩阵
    L = sympy.simplify(M.T*M)
    L_1 = L**(-1)
    L_1 = L_1.evalf(subs={e1:0,e2:0,e3:0})
    y = Matrix(Ri)
    error_i = y-M*beta_m.T
    var_i = sympy.simplify(1/(N-p-1)*error_i.T*error_i)
    std_error_alpha = var_i[0,0]*L_1[0,0]
    
    #计算alpha的t统计量
    y_bar = np.mean(Ri)
    m_bar = np.mean(m_ad)
    s_bar = np.mean(s_ad)
    h_bar = np.mean(h_ad)
    t_alpha = abs(y_bar-MKT_1*m_bar-SMB_1*s_bar-HML_1*h_bar)*std_error_alpha**(-0.5)
    t_alpha = sympy.simplify(t_alpha)
    
    #定义最小值函数
    f_talpha=sympy.lambdify(([e1,e2,e3]),t_alpha)
    
    def f_a(x):
        e1=x[0]
        e2=x[1]
        e3=x[2]
        return f_talpha(e1,e2,e3)    
    
    def con(args):
        # 约束条件 分为eq 和ineq
        # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0  
        alpha,beta1,beta2,beta3 = args
        cons = ({'type': 'ineq', 'fun': lambda x: 1e-8 - x[0]**2 - x[1]**2 - x[2]**2},\
                 {'type': 'eq', 'fun': lambda x:  alpha + x[0]*beta1 + x[1]*beta2 
                  + x[2]*beta3})
        return cons
    
    args1 = (alpha_1, MKT_1, SMB_1, HML_1)  #alpha,beta1,beta2,beta3
    cons = con(args1)
    
    
    e_x = optimize.minimize(f_a,x0=[0,0,0],constraints=cons)
    e_ad=e_x.x
    
    fac_mat_ad = pd.DataFrame({'Ri_ad':Ri,
                               'MKT_ad':MKT-np.ones(len(Rm))*e_ad[0],
                               'SMB_ad':SMB-np.ones(len(Rm))*e_ad[1],
                               'HML_ad':HML-np.ones(len(Rm))*e_ad[2]})
    
    # 计算沪深300 alpha & beta(三因子模型)调整
    model_se_ad = smf.ols('Ri_ad ~ MKT_ad + SMB_ad + HML_ad', data=fac_mat_ad)
    result_se_ad = model_se_ad.fit()  # 拟合
    #result_se_ad.summary()
    alpha_series_ad.append(result_se_ad.params[0])
    
# 登出系统
bs.logout()

#计算年化alpha
alpha_series_y = [a*12 for a in alpha_series_3]
alpha_series_ad_y = [a*12 for a in alpha_series_ad]

plt.figure(figsize=(15,10))
plt.plot(range(1,len(alpha_series_3)+1),alpha_series_y,color='red',marker='o',label='沪深300指数年化Alpha')
plt.plot(range(1,len(alpha_series_3)+1),alpha_series_ad_y,color='blue',marker='o',label='沪深300指数年化Alpha（调整）')
plt.xlabel(u"一年滚动数据",fontsize=13)
plt.ylabel(u"Alpha",rotation=90,fontsize=13)
plt.title(u"沪深300指数年化Alpha")
plt.xticks([(12*i+1) for i in range(0,6)],[date_count[12*i] for i in range(1,7)])
plt.legend(loc=2)
plt.show()