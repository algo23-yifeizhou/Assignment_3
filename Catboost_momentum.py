#%%导入包
import pickle
from time import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd 
import numpy as np

import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
import numba 
from numba import jit
import warnings
import matplotlib 
warnings.filterwarnings('ignore')  #过滤代码运行过程中烦人的警告
matplotlib.rcParams['axes.unicode_minus']=False #解决画图中负数显示问题

#%%通用函数
from scipy import stats
#计算LLT
import numba
@numba.jit 
def shift(self, num, fill_value=np.nan):
    result = np.empty_like(self)
    if num > 0:
        result[:num] = fill_value
        result[num:] = self[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = self[-num:]
    else:
        result[:] = self
    return result

@numba.jit
def llt2(values, window):
    a   = 2/(1+window); p0 = a-a*a/4; p1 = a*a/2;
    p2  = a-a*a*3/4;f1 = 2*(1-a); f2 = (1-a)*(1-a)
    val = np.array(values); out = np.array(values)
    tmp = (p0*val)+(p1*shift(val,1))-(p2*shift(val,2))
    for i in np.arange(2,val.shape[0]):
        out[i] = tmp[i] + f1*out[i-1] - f2*out[i-2]
    out[0:window-1] = np.nan
    return out
def Indicators(value):
    '''
    PARAMETERS
    value:t级别净值序列,含初始净值
    hold_freq:t级别（调仓周期）
    t_days:交易日序列,只用于计算跨月收益集中度t_hhi,len(t_days) = len(value) - 1
    
    RETURN 
    result:策略各项指标,可自行增删
    '''
    from scipy.stats import skew, kurtosis, norm
    #盈亏比
    def PlRatio(value):
        value = value[1:]-value[:-1]
        ratio = -value[value>0].mean()/value[value<0].mean()
        return ratio
    #日胜率,日基准收益为0
    def WinRate(Returns):
        pos = sum(Returns > 0)
        neg = sum(Returns < 0)
        return pos/(pos+neg)
    #最大回撤
    def MaxDrawBack(value):
        i = np.argmax(np.maximum.accumulate(value)-value)  # 结束位置
        if i == 0:
            return 0
        j = np.argmax(value[:i])  # 开始位置
        return (value[j]-value[i])/(value[j])
    #最长回撤时间，水下时间
    def MaxDrawBackDays(value):
        maxValue = 0
        maxValueIndex = []
        for k,v in enumerate(value):
            if v >= maxValue:
                maxValue = v
                maxValueIndex.append(k)
        last = len(value)-maxValueIndex[-1]-1 #回测最后处于最长回撤时期
        if len(maxValueIndex) == 1: #未创新高
            return last
        else:
            maxValueIndex = pd.Series(maxValueIndex)
            maxValueIndex -= maxValueIndex.shift(1) 
            maxValueIndex = maxValueIndex.dropna().values
            return max(maxValueIndex.max(),last) 
    #下行波动率
    def UnderVo(Returns):
        sigma = 0
        num = len(Returns)
        for k,r in enumerate(Returns):
            rMean = np.mean(Returns[:k])
            if r < rMean:
                sigma += (r-rMean)**2
        sigma = np.sqrt(sigma*250/num)
        return sigma
    
    #收益集中度  ret：收益率序列Series，index=date
    def getHHI(ret): 
        if ret.shape[0]<=2:
            return np.nan
        weight=ret/ret.sum()
        hhi=(weight**2).sum()
        hhi=(hhi-ret.shape[0]**-1)/(1.-ret.shape[0]**-1)
        return hhi
    #+/-/跨月收益集中度 
    def ReturnsConcentration(ret):
        pos_ret = ret[ret>0]
        neg_ret = ret[ret<0]
        pos_hhi = getHHI(pos_ret) # concentration of positive returns per bet
        neg_hhi = getHHI(neg_ret) # concentration of negative returns per bet
        t_hhi = getHHI(ret.groupby(pd.TimeGrouper(freq='M')).count()) # concentr. bets/month
        return pos_hhi,neg_hhi,t_hhi
    
    #PSR #ret: 1darray,1d收益率序列threshold:夏普率参照 rf: 年化无风险收益率
    def calcPsr(ret,sharpe,threshold=0,rf=0): 
        skw = skew(ret)
        kur = kurtosis(ret,fisher=False) #fisher=False:正态分布峰度=3
        prob = norm.cdf(((sharpe-threshold)*np.sqrt(ret.shape[0]))/np.sqrt(1-skw*sharpe+0.25*(kur-1)*sharpe**2))
        return prob #夏普率大于基准的概率
    #计算
    value=np.array(value)
    value=value/value[0] #每日净值 1darray
    value1=pd.Series(value)
    Returns=value1.pct_change(1).dropna().values #每日收益率 1darray
    ###
    TotalRetn = round(value[-1]*100-100,2) #总收益
    AnnualRetn= round(pow(value[-1],250/(len(value[1:])))*100-100,2) #年化收益
    Plr = round(PlRatio(value),2) #盈亏比
    Wr  = round(WinRate(Returns)*100,2) #日胜率
    Volatility     = round(np.sqrt(Returns.var()*250)*100,2) #年化波动率
    SharpRatio     = round((AnnualRetn)/Volatility,2) #年化夏普比
    # PSR
    PSRatio        = round(calcPsr(ret=Returns,sharpe=SharpRatio,threshold=1,rf=0),2) #概率夏普比 
    ###
    MaxDrawback    = round(MaxDrawBack(value)*100,2) #最大回撤
    KMRatio        = round(AnnualRetn/MaxDrawback,2) #卡玛比率
    MaxDrawbackDays= int(MaxDrawBackDays(value)) #最长回撤时间
    SortinoRatio   = round((AnnualRetn-4)/UnderVo(Returns)/100,2) #索提诺比率
    # HHI
#     Returns_s      = pd.Series(Returns)
#     Returns_s.index  = pd.to_datetime(t_days) 
#     HHI = getHHI(ret = Returns_s) #收益集中度
#     pos_hhi,neg_hhi,t_hhi = ReturnsConcentration(ret = Returns_s)
    ###
    Returns2 = np.sort(Returns)
    Max2  = round(Returns2[0]*100,2) #最大单日回撤
    var5  = round(Returns2[int(len(Returns2)*0.05)]*100,2) #收益率5%分位数
    '''
    以下输出值按需增删
    '''
    columns=['总收益','年化收益','波动率','夏普比','最大回撤','日胜率','盈亏比','最长回撤日数','calmar比率','单期最大回撤']
    data=[[TotalRetn,AnnualRetn,Volatility,SharpRatio,MaxDrawback,Wr,Plr,MaxDrawbackDays,KMRatio,Max2]]
    result=pd.DataFrame(columns=columns,data=data)
    return result

    # 画净值曲线
def net_value(returns, benchmark=None,commission=0.0000):
    dates = returns.index
    net_value_se = pd.Series(((1-commission)*(1 + returns.values)).cumprod(),index=dates,name='intraday')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=(16,9))
    ax1 = plt.axes()
    ax1.set_xlabel('date')
    ax1.set_ylabel('net value')
    ax1.set_title(net_value_se.name + '  Net Value')
    # 净值曲线
    ax1.plot(net_value_se,linestyle='-',label='Net value')
    # jis
    graph_bottom = plt.ylim()[0]
    graph_top = plt.ylim()[1]
    graph_height = graph_top - graph_bottom
    
    indi = Indicators(net_value_se)
    indi =indi.rename(index={0:net_value_se.name})
    if benchmark is not None:

        # benchmark净值曲线
        ax1.plot(benchmark_net_value,linestyle='-.',color='grey',label=bench_code)
        graph_bottom = plt.ylim()[0]
        graph_top = plt.ylim()[1]
        graph_height = graph_top - graph_bottom
    # 超额收益曲线
        excess = net_value_se.pct_change().dropna() - benchmark_net_value.pct_change().dropna()
        ax1.bar(x=excess.index,height=(0.2*graph_height/excess.min())*excess.values,bottom=1,color='orange',label='Excess return rate')
        indi_benchmark = Indicators(benchmark_net_value)
        indi_benchmark =indi_benchmark.rename(index={0:bench_code})
        indi = indi.append(indi_benchmark)
    #日亏损
    rt = net_value_se.pct_change().dropna()
    rt[rt>0]=0
    drawdown_se = rt
    ax1.bar(x=drawdown_se.index,height=(-0.2*graph_height/rt.min())*drawdown_se.values,bottom=graph_top,color='silver',label='Drawdown')
    
    
    plt.legend()
    plt.show()
    return indi

#%% 读取清洗后的数据
# shaped_data 格式
# [第一天数据，第二天数据，。。。]
# 其中，第一天数据=np.array([future_arr, future_volume_arr, open_interest_arr,
# #                         index_arr, index_volume_arr,
# #                         basis_arr]).T
folder_path = 'my_data'
def read_pkl_data(code_name, folder_path):
    data_path = folder_path + '\\{}_shaped_data.pkl'.format(code_name)
    time_path = folder_path + '\\time_info.pkl'
    with open(data_path, 'rb') as f:
        shaped_data = pickle.load(f)
    with open(time_path, 'rb') as f_t:
        time_info = pickle.load(f_t)

    return shaped_data, time_info

IC_shaped_data, time_info = read_pkl_data('IC',folder_path)
IH_shaped_data, _         = read_pkl_data('IH',folder_path)
IF_shaped_data, _         = read_pkl_data('IF',folder_path)

date_list = time_info[0]
time_list = time_info[1]

#%% 计算区间收益率
def get_grouped_data(shaped_data,freq,LLT_w_fast=10, LLT_w_slow=30,back_mltp=0,LLT_udly_idx=0):
    '''
    shaped_data: 清洗好的源数据,昨天和今天的数据拼接,(239,240,241)=(y_15:00,t_9:30,t_9:31)
    freq: 区间长度/采样频率
    LLT_udly_idx: 计算LLT的源数据所在位置索引: 期货->0, 指数->3
    LLT_w_fast: 
    LLT_w_slow: 
    back_mltp: times of the freq when look back,回看昨天的段数,为0:前一天收盘价为第一个数据
    '''
    # slicer = slice(239-back_mltp*freq, 480, freq)
    # selected_time = time_list[slicer]
    # data_daily_raw = shaped_data[t]

    slicer_t = slice(240, 481, freq)
    slicer_y = slice(239-back_mltp*freq, 240, freq)
    slicers = (slicer_y, slicer_t)
    selected_time = time_list[slicer_y] + time_list[slicer_t]

    # data_daily_raw是昨天和今天的数据拼接
    # #     data_daily_raw = np.array([future_arr, future_volume_arr, open_interest_arr,
    # #                             index_arr, index_volume_arr,
    # #                             basis_arr])
    def calc_daily_pcts(data_daily, slicers):
    # data_daily是daily_data_raw 添加了LLT滤波数据
    # #     data_daily = np.array([future_arr, LLT_w_fast, LLT_w_slow,
    #                              future_volume_arr, open_interest_arr,
    # #                            index_arr, index_volume_arr,
    # #                            basis_arr])
        slicer_y, slicer_t = slicers
        data_t = data_daily[slicer_t]
        data_y = data_daily[slicer_y]
        selected_data = np.concatenate((data_y,data_t),axis=0)
        pct_change = selected_data[1:] / selected_data[:-1] - 1
        return  pct_change

    from tqdm import tqdm
    grouped_data = []
    for _, data_daily_raw in tqdm(enumerate(shaped_data)):
        data_daily = np.insert(data_daily_raw,1,llt2(data_daily_raw[:,LLT_udly_idx],LLT_w_slow),axis=1)
        data_daily = np.insert(data_daily,1,llt2(data_daily_raw[:,LLT_udly_idx],LLT_w_fast),axis=1)
        grouped_data.append(calc_daily_pcts(data_daily, slicers))

    grouped_arr = np.array(grouped_data)
    # Nan 和 inf 出现在成交量列，原因在于涨停后成交量为零，计算式除以0了产生了nan和inf,将其替换为0即可
    grouped_arr[np.isnan(grouped_arr)] = 0.0
    grouped_arr[np.isinf(grouped_arr)] = 0.0
    # grouped_arr格式为三维的array, (date,(r1,r2...rn),(future,LLT_f_fast,LLT_f_slow,fut_vlm,fut_oi,index,idx_vlm,basis))
    return grouped_arr,selected_time[:-1]

freq_sig = 30
freq_pre = 30
LLT_w_fast = 10
LLT_w_slow = 30
back_mltp = 0
IF_grouped_arr_sig,selected_time = get_grouped_data(IF_shaped_data,freq_sig,LLT_w_fast, LLT_w_slow,back_mltp=back_mltp,LLT_udly_idx=0)
IH_grouped_arr_sig,selected_time = get_grouped_data(IH_shaped_data,freq_sig,LLT_w_fast, LLT_w_slow,back_mltp=back_mltp,LLT_udly_idx=0)
IC_grouped_arr_sig,selected_time = get_grouped_data(IC_shaped_data,freq_sig,LLT_w_fast, LLT_w_slow,back_mltp=back_mltp,LLT_udly_idx=0)

IF_grouped_arr_pre,selected_time = get_grouped_data(IF_shaped_data,freq_pre,LLT_w_fast, LLT_w_slow,back_mltp=back_mltp,LLT_udly_idx=0)
IH_grouped_arr_pre,selected_time = get_grouped_data(IH_shaped_data,freq_pre,LLT_w_fast, LLT_w_slow,back_mltp=back_mltp,LLT_udly_idx=0)
IC_grouped_arr_pre,selected_time = get_grouped_data(IC_shaped_data,freq_pre,LLT_w_fast, LLT_w_slow,back_mltp=back_mltp,LLT_udly_idx=0)

from collections import namedtuple
Strgy_args = namedtuple('Strgy_args',['orientation','threshold','commissions',
                                      'signal_frequency','predict_frequency'])

Criteria = namedtuple('Criteria',['t_arr', 'p_OLS_arr','ann_retn_arr','IC_arr','WinRate_arr'])


strgy_args = Strgy_args(orientation='momentum',# 方向：动量momentum, 反转reverse
                        threshold=0, # 交易判断临界值
                        commissions=0,
                        signal_frequency=freq_sig,
                        predict_frequency=freq_pre)

#%% 设置训练集和测试集
import catboost
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor,CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score
# grouped_arr格式为三维的array, (date,(r1,r2...rn),(future,LLT_f_fast,LLT_f_slow,fut_vlm,fut_oi,index,idx_vlm,basis))
signal_arr = IH_grouped_arr_sig[:,:,0]
train_slicer = slice(0,1200)
test_slicer = slice(1200,signal_arr.shape[0])

train_future_set = IH_grouped_arr_sig[train_slicer,:,0]
train_future_set = np.concatenate((train_future_set,IC_grouped_arr_sig[train_slicer,:,0]),axis=0)
train_future_set = np.concatenate((train_future_set,IF_grouped_arr_sig[train_slicer,:,0]),axis=0)

test_future_IH = IH_grouped_arr_pre[test_slicer,:,0]
test_future_IC = IC_grouped_arr_pre[test_slicer,:,0]
test_future_IF = IF_grouped_arr_pre[test_slicer,:,0]
#%% 打分类标签
def get_label_and_features(df,label,features):
    LnF = pd.DataFrame() # label and features
    func = lambda x: 1 if x>0 else (-1 if x<0 else np.NaN)
    LnF[features] = df[features].applymap(func)
    LnF[label] = df[label].apply(func)
    LnF = LnF.dropna(axis=0)
    return LnF.astype(int)

columns = np.array(['r{}'.format(x) for x in range(train_future_set.shape[1])])
train_df = pd.DataFrame(train_future_set, columns=columns)

IH_test_df = pd.DataFrame(test_future_IH, columns=columns)
IC_test_df = pd.DataFrame(test_future_IC, columns=columns)
IF_test_df = pd.DataFrame(test_future_IF, columns=columns)

label = 'r7'
features = ['r1','r5','r6']
###################不打分类标签这段注释掉######################
train_set = get_label_and_features(train_df,label,features)
IH_test_set = get_label_and_features(IH_test_df,label,features)
IC_test_set = get_label_and_features(IC_test_df,label,features)
IF_test_set = get_label_and_features(IF_test_df,label,features)
#############################################################
#%%
y = train_set[label]
X = train_set[features]

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
X_test = IH_test_set
categorical_features_indices = np.where(X.dtypes != float)[0]
# %% Choose the best param and early stop
classify_params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'eval_metric': metrics.Accuracy(),
    'random_seed': 23,
    'logging_level': 'Silent',
    'use_best_model': False
}

train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)
model = CatBoostClassifier(**classify_params)

model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_validation, y_validation),
    logging_level='Verbose',  # you can uncomment this for text output
    plot=True
)
#%%
cv_params = model.get_params()
cv_params.update({
    'loss_function': metrics.Logloss()
})
cv_data = cv(
    Pool(X, y, cat_features=categorical_features_indices),
    cv_params,
    plot=True
)
#%%
# predictions = model.predict(X_test)
# # predictions_probs = model.predict_proba(X_test)
# print(predictions)
# # print(predictions_probs)



backtest_set = IC_test_set
back_test_df = IC_test_df
retns_ML = []
retns = []
for t, row in backtest_set.iterrows():
    signal_ML = model.predict(row)
    r = back_test_df['r7'][t]
    if signal_ML == 1:
        retns_ML.append(r)
    elif signal_ML == 0:
        retns_ML.append(-1 * r)
    else:
        retns_ML.append(0)
    
    if back_test_df['r1'][t] > 0:
        retns.append(r)
    elif back_test_df['r1'][t] < 0:
        retns.append(-1 * r)
    else:
        retns.append(0)

# retns_ML_se = pd.Series(retns_ML,index=date_list[test_slicer])
# retns_se = pd.Series(retns,index=date_list[test_slicer])
retns_ML_se = pd.Series(retns_ML)
retns_se = pd.Series(retns)
net_value(retns_ML_se, benchmark=None,commission=0.0000)
#%%
net_value(retns_se, benchmark=None,commission=0.0000)

# %%
