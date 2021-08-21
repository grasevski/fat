import os
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib as plt
import math
import time
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import os
import talib as ta
from sklearn.preprocessing import MinMaxScaler
yesterday = '20210630'
today='20210701'
tmrw= '20210702'
ts.set_token('deeed5c0b53302fc47d6f00400c7a42941b2f7f32bf2b0cdd445421d')
pro = ts.pro_api()

def get_data(ts_code, start_date,end_date, retry = 10, pause = 20):
    for _ in range(retry):
        try:
            df_new_basic = ts.pro_bar(ts_code= ts_code, adj='qfq',start_date=start_date, end_date=end_date)
            df_new_info = pro.daily_basic(ts_code= ts_code, fields='trade_date,turnover_rate,pe',start_date=start_date, end_date=end_date)
            column = 'trade_date'
            df_new = df_new_basic.join(df_new_info.set_index(column), on=column)
        except:
            print('Timed Out ... Reconnectiong ...')
            time.sleep(pause)
        else:
            return df_new

directory = r'D:/Trading_Bot/Data/Individual_Stock/'
for filename in os.listdir(directory):
    if filename.endswith(".csv") and len(filename) == 13:
        csv_file = os.path.join(directory, filename)
        csv_name = csv_file.replace(".csv", "")
        d='D:/Trading_Bot/Data/Individual_Stock/'
        csv_name = csv_name.replace(d,"")
        df = pd.read_csv(csv_file)
        if str(df.iloc[-1, df.columns.get_loc('trade_date')]) != yesterday and str(df.iloc[-1, df.columns.get_loc('trade_date')]) != today:
            df_new = get_data(ts_code=csv_name, start_date='20140101', end_date=tmrw)
            df = df_new.iloc[::-1].reset_index(drop=True)
            df = df.fillna(0)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            outputpath = "D:/Trading_Bot/Data/Individual_Stock/"+csv_name+".csv"
            df.to_csv(outputpath,sep=',',index=False,header=True)
            print(csv_name + ' gathered')
        elif str(df.iloc[-1, df.columns.get_loc('trade_date')]) != today:
            df_new = get_data(ts_code=csv_name, start_date=today, end_date=tmrw)
            df = df.append(df_new, ignore_index = True)
            df = df.reset_index(drop=True)
            df = df.fillna(0)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            outputpath = "D:/Trading_Bot/Data/Individual_Stock/"+csv_name+".csv"
            df.to_csv(outputpath,sep=',',index=False,header=True)
            print(csv_name + ' updated')
        elif len(filename)!=13:
            csv_file =os.path.join(directory, filename)
            os.remove(csv_file)
            print(csv_name + ' is removed')
        else:
            print(csv_name + ' already updated')
##            
##WR:上升通道没有破线的稳涨的可能性大，但是涨多少得看
##WR_L[i]>40 and \
##macd[i]<0 and \
##ma10[i]>ma20[i] and\
##ma20[i]>ma55[i] and ma55[i]>ma144[i]
##
##ma5[i]>=ma10[i] and\
##           ma20[i]>0.98 and\
##           ma20[i]<1.05 and\
##           ma5[i]<=ma20[i]
##ma5[i]<=ma20[i] and\
##           ma20[i]>=ma55[i] and\
##           ma20[i]>0.98 and\
##           ma20[i]<1.05


##DTPL:

##DTPLMA20:
##ma20[i]>1 and \
##ma5[i]>ma10[i] and ma10[i]>ma20[i] and ma20[i]>ma55[i]

##28.180%,0.344%


## 01/05/2021:
##        ma10[i]>=ma20[i] and\
##           ma20[i]>=ma55[i] and\
##           PSAR_S[i]<3 and\
##           PSAR_S[i]>-8 and\
def get_label ():
    label = [0,0]
    for i in range(2,(len(closing_price))-2,1):
        if amount[i]>400000:
            if amount[i] > 1000000:
                if ((high[i+2]-open_l[i+1])/open_l[i+1])>=0.05:
                    label.append(1)
                else:
                    label.append(0)
            else:
                if amount[i] > amount[i-1]:
                    if ((high[i+2]-open_l[i+1])/open_l[i+1])>=0.05:
                        label.append(1)
                    else:
                        label.append(0)
                else:
                    label.append(0)
        else:
            label.append(0)
    label.append(0)
    label.append(0)
    return label


def get_data_label ():
    data_label = [0,0]
    for i in range(2,(len(closing_price)),1):
        if amount[i]>400000:
            if amount[i] > 1000000:
                data_label.append(1)
            else:
                if amount[i] > amount[i-1]:
                    data_label.append(1)
                else:
                    data_label.append(0)
        else:
            data_label.append(0)
    return data_label


def actual_return():
    actual_return=[]
    for i in range(len(closing_price)-2):
        re = (high[i+2]-open_l[i+1])/open_l[i+1]
        if re>=0.05:
            ret = 0.05
        else:
            ret = (closing_price[i+2]-open_l[i+1])/open_l[i+1]
        actual_return.append(ret)
    actual_return.append(0)
    actual_return.append(0)
    return actual_return

                   
#def RSI():
#    RSI=[0,0,0,0,0,0,0,0]
#    l=(2/(8+1))
#    for i in range(8, (len(closing_price)),1):
#        U=[]
#        D=[]
#        for a in range(i-7,i,1):
#            if (closing_price[a]-closing_price[a-1])>0:
#                u=(closing_price[a]-closing_price[a-1])/closing_price[a-1]
#                U.append(u)
#            else:
#                d=(closing_price[a-1]-closing_price[a])/closing_price[a-1]
#                D.append(d)
#        if len(D)==0:
#            AvgD = 0
#            RS=100
#        elif len(U) == 0:
#            AvgU= 0
#            RS = 0
#        else:
#            AvgU=sum(U)/len(U)
#            AvgD=sum(D)/len(D)
#            RS=100-(100/(1+(AvgU/AvgD)))
#        RSI.append(RS)
#    return RSI
#
#def Vol_MA5 ():
#    Vol_MA5 =[0,0,0,0]
#    for i in range(4, (len(closing_price)),1):
#        s=((volume[i-4]+volume[i-3]+volume[i-2]+volume[i-1]+volume[i])/5)*100/volume[i]
#        Vol_MA5.append(s)
#    return Vol_MA5
#
#def Vol_MA10 ():
#    Vol_MA10 =[0,0,0,0,0,0,0,0,0]
#    for i in range(9, (len(closing_price)),1):
#        s=((volume[i-9]+volume[i-8]+volume[i-7]+volume[i-6]+volume[i-5]+volume[i-4]+volume[i-3]+volume[i-2]+volume[i-1]+volume[i])/10)*100/volume[i]
#        Vol_MA10.append(s)
#    return Vol_MA10
#
#def Vol_def ():
#    Vol_def =[]
#    for i in range(0,(len(closing_price)),1):
#        d= Vol_MA10_L[i]-Vol_MA5_L[i]
#        Vol_def.append(d)
#    return Vol_def
#
#def MA5 ():
#    MA5 =[0,0,0,0]
#    for i in range(4, (len(closing_price)),1):
#        s=((closing_price[i-4]+closing_price[i-3]+closing_price[i-2]+closing_price[i-1]+closing_price[i])/5)*100/closing_price[i]
#        MA5.append(s)
#    return MA5
#
#def MA5_def():
#    MA5_def =[]
#    for i in range(0,(len(closing_price)),1):
#        d= 1-MA5_L[i]
#        MA5_def.append(d)
#    return MA5_def
#
#def MA10 ():
#    MA10 =[0,0,0,0,0,0,0,0,0]
#    for i in range(9, (len(closing_price)),1):
#        s=((closing_price[i-9]+closing_price[i-8]+closing_price[i-7]+closing_price[i-6]+closing_price[i-5]+closing_price[i-4]+closing_price[i-3]+closing_price[i-2]+closing_price[i-1]+closing_price[i])/10)*100/closing_price[i]
#        MA10.append(s)
#    return MA10
#
#def MA10_def():
#    MA10_def =[]
#    for i in range(0,(len(closing_price)),1):
#        d= 1-MA10_L[i]
#        MA10_def.append(d)
#    return MA10_def
#
#def EMA8():
#    EMA8 = [0,0,0,0,0,0,0]
#    s=(closing_price[0]+closing_price[1]+closing_price[2]+closing_price[3]+closing_price[4]+closing_price[5]+closing_price[6]+closing_price[7])/8
#    EMA8.append(s)
#    k=(2/(8+1))
#    for i in range(8, (len(closing_price)),1):
#        e=closing_price[i]*k+EMA8[i-1]*(1-k)
#        EMA8.append(e)
#    return EMA8
#
#def EMA12():
#    EMA12 = [0,0,0,0,0,0,0,0,0,0,0]
#    s=(closing_price[0]+closing_price[1]+closing_price[2]+closing_price[3]+closing_price[4]+closing_price[5]+closing_price[6]+closing_price[7]+closing_price[8]+closing_price[9]+closing_price[10]+closing_price[11])/12
#    EMA12.append(s)
#    k=(2/(12+1))
#    for i in range(12, (len(closing_price)),1):
#        e=closing_price[i]*k+EMA12[i-1]*(1-k)
#        EMA12.append(e)
#    return EMA12
#
#def MACD ():
#    MACD=[]
#    for i in range(0, (len(closing_price)),1):
#        a=EMA8_L[i]
#        b=EMA12_L[i]
#        CD=(a-b)/closing_price[i]
#        MACD.append(CD)
#    return MACD
#
#def EMA5_MACD():
#    EMA5 = [0,0,0,0]
#    s=(MACD_L[0]+MACD_L[1]+MACD_L[2]+MACD_L[3]+MACD_L[4])/5
#    k=(2/(5+1))
#    EMA5.append(s)
#    for i in range(5, (len(MACD_L)),1):
#        e=MACD_L[i]*k+EMA5[i-1]*(1-k)
#        EMA5.append(e)
#    return EMA5
#
#def MACD_def():
#    MACD_def =[]
#    for i in range(0,(len(closing_price)),1):
#        d= MACD_L[i]- MACD_signal_L[i]
#        MACD_def.append(d)
#    return MACD_def
#
#def BOL_UP():
#    BOL_UP = [0,0,0,0,0,0,0,0,0]
#    for i in range(9, (len(closing_price)),1):
#        s=closing_price[i-9]+closing_price[i-8]+closing_price[i-7]+closing_price[i-6]+closing_price[i-5]+closing_price[i-4]+closing_price[i-3]+closing_price[i-2]+closing_price[i-1]+closing_price[i]
#        m=s/10
#        su=[]
#        for n in range(i-9,i,1):
#            a=closing_price[n]-m
#            b=a*a
#            su.append(b)
#        sun=sum(su)
#        sund=sun/9
#        std = math.sqrt(sund)
#        BOL= (m+(1.96*std))*100/closing_price[i]
#        BOL_UP.append(BOL)
#    return BOL_UP
#
#def BOL_UP_def():
#    BOL_UP_def =[]
#    for i in range(0,(len(closing_price)),1):
#        d= BOL_UP_L[i] - 100
#        BOL_UP_def.append(d)
#    return BOL_UP_def
#
#def BOL_LOW():
#    BOL_LOW = [0,0,0,0,0,0,0,0,0]
#    for i in range(9, (len(closing_price)),1):
#        s=closing_price[i-9]+closing_price[i-8]+closing_price[i-7]+closing_price[i-6]+closing_price[i-5]+closing_price[i-4]+closing_price[i-3]+closing_price[i-2]+closing_price[i-1]+closing_price[i]
#        m=s/10
#        su=[]
#        for n in range(i-9,i,1):
#            a=closing_price[n]-m
#            b=a*a
#            su.append(b)
#        sun=sum(su)
#        sund=sun/9
#        std = math.sqrt(sund)
#        BOL= (m-1.96*std)*100/closing_price[i]
#        BOL_LOW.append(BOL)
#    return BOL_LOW
#
#def BOL_LOW_def():
#    BOL_LOW_def =[]
#    for i in range(0,(len(closing_price)),1):
#        d= 100 - BOL_LOW_L[i]
#        BOL_LOW_def.append(d)
#    return BOL_LOW_def
#
#def get_month():
#    trade_date = []
#    for date in original_data['trade_date']:
#        dt = str(date)
#        dtu = dt[4:-2]
#        trade_date.append(dtu)
#    return trade_date
#
#items = {'trade_date':[], 'open':[], 'high':[], 'low':[], 'close':[], 'pct_chg':[], 'vol':[], 'amount':[], 'Vol_MA5':[], 'Vol_MA10':[], 'Vol_def':[], 'MA5':[], 'MA10':[], 'MA5_def':[], 'MA10_def':[], 'MACD':[], 'MACD_signal':[], 'MACD_def':[], 'BOL_UP_def':[], 'BOL_UP':[], 'BOL_LOW_def':[], 'BOL_LOW':[], 'RSI':[], 'Label':[]}
#dtest = pd.DataFrame (items, columns = ['trade_date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount', 'Vol_MA5', 'Vol_MA10', 'Vol_def', 'MA5', 'MA10', 'MA5_def', 'MA10_def', 'MACD', 'MACD_signal', 'MACD_def', 'BOL_UP_def', 'BOL_UP', 'BOL_LOW_def', 'BOL_LOW', 'RSI', 'Label'])
#
#items = {'trade_date':[], 'open':[], 'high':[], 'low':[], 'close':[], 'pct_chg':[], 'vol':[], 'amount':[], 'Vol_MA5':[], 'Vol_MA10':[], 'Vol_def':[], 'MA5':[], 'MA10':[], 'MA5_def':[], 'MA10_def':[], 'MACD':[], 'MACD_signal':[], 'MACD_def':[], 'BOL_UP_def':[], 'BOL_UP':[], 'BOL_LOW_def':[], 'BOL_LOW':[], 'RSI':[], 'Label':[]}
#aggregate_data = pd.DataFrame (items, columns = ['trade_date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount', 'Vol_MA5', 'Vol_MA10', 'Vol_def', 'MA5', 'MA10', 'MA5_def', 'MA10_def', 'MACD', 'MACD_signal', 'MACD_def', 'BOL_UP_def', 'BOL_UP', 'BOL_LOW_def', 'BOL_LOW', 'RSI', 'Label'])
def EMA(lst,n):
    lst = pd.Series(lst)
    modPrice = lst.copy()
    sman=modPrice.rolling(n).mean()
    modPrice.iloc[0:n] = sman[0:n]
    ema = modPrice.ewm(span=n, adjust=False).mean()
    return ema

def PSAR (AFrate):
    direction = [None,None,None]
    psar =[None,None,None]
    EP = [None,None,None]
    pri_EP_count = 1
    AF = min(pri_EP_count*AFrate,0.2)
    if high[4]> high[3]:
        psar.append(high[3])
        EP.append(high[3])
        direction.append('F')
    else:
        psar.append(low[3])
        EP.append(low[3])
        direction.append('F')
    for i in range(4,len(low)):
        if low[i]<psar[i-1] and direction[i-1] == 'R':
            direction.append('F')
            AF = 0.02
            npsar = EP[i-1] - AF*(EP[i-1] - psar[i-1])
            psar.append(round(npsar,2))
            pri_EP_count = 1
            if low[i]< EP[i-1]:
                EP.append(low[i])
            else:
                EP.append(EP[i-1])
        elif high[i]<=psar[i-1] and direction[i-1] == 'F':
            if low[i]< EP[i-1]:
                pri_EP_count = pri_EP_count + 1
                EP.append(low[i])
            else:
                EP.append(EP[i-1])
            direction.append('F')
            AF = min(pri_EP_count*AFrate,0.2)
            npsar = psar[i-1] - AF*(psar[i-1] - min(low[i-4],low[i-3],low[i-2],low[i-1],low[i]))
            psar.append(round(npsar,2))
        elif low[i]>=psar[i-1] and direction[i-1] == 'R':
            if high[i]> EP[i-1]:
                pri_EP_count = pri_EP_count + 1
                EP.append(high[i])
            else:
                EP.append(EP[i-1])
            direction.append('R')
            AF = min(pri_EP_count*AFrate,0.2)
            npsar = psar[i-1] + AF*(max(high[i-1],high[i]) - psar[i-1])
            psar.append(round(npsar,2))
            
        elif high[i]> psar[i-1] and direction[i-1] == 'F':
            direction.append('R')
            AF = 0.02
            npsar = low[i-1]
            psar.append(round(npsar,2))
            pri_EP_count = 1
            if high[i]> EP[i-1]:
                EP.append(high[i])
            else:
                EP.append(EP[i-1])
        else:
            print('DIRECTION_ERROR')
    return psar


def DMA(series,weight):
    a = series[0]
    Y = [a]
    for i in range(1,len(series)):
        b = weight[i]*series[i] + (1-weight[i])*Y[i-1]
        Y.append(b)
    return Y

def MACD(series, short = 12, long = 26, mid = 9):
    DIF = ta.EMA(series,short)-ta.EMA(series,long)
    DEA = ta.EMA(DIF,mid)
    MACD = (DIF-DEA)*2
    return DIF, DEA, MACD

def WRF(n = 6):
    v1 = []
    for i in range(len(closing_price)):
        h_l = [h for h in list(high[max(0,i+1-n):i+1])]
        h_n = np.float(max(h_l))
        l_l = [l for l in list(low[max(0,i+1-n):i+1])]
        l_n = np.float(min(l_l))
        a = np.float(h_n - l_n)
        b = np.float(h_n - closing_price[i])
        if a != 0:
            c = np.float(b/a)
            d = c*100
        else:
            d = 0
        v1.append(d)
    return pd.Series(v1)

def MCST():
    turn_per = turn/100
    L = DMA(av_price,turn_per)
    return L

def _ma(series, n):
    """
    移动平均
    """
    return series.rolling(n).mean()

def CYHT(V_period = 34, E_period = 13):
    var2 =low.rolling(V_period).min()
    var3 =high.rolling(V_period).max()
    var1 = (2*closing_price+high+low+open_l)/5
    SK =ta.EMA((((var1-var2)/(var3-var2))*100),E_period)
    SD = ta.EMA(SK,3)
    return SK, SD

def CJDX ():
    Var1 = (2*closing_price+high+low)/4
    Var2 = (4*Var1+3*Var1.shift(1)+2*Var1.shift(2)+Var1.shift(3))/10
    Var3 = (4*Var2+3*Var2.shift(1)+2*Var2.shift(2)+Var2.shift(3))/10
    Var4 = (4*Var3+3*Var3.shift(1)+2*Var3.shift(2)+Var3.shift(3))/10
    J = (Var4 - Var4.shift(1))*100/Var4.shift(1)
    D = J.rolling(3).mean()
    return J, D


def vosc(df, n=12, m=26):
    """
    成交量震荡	vosc(12,26)
    VOSC=（MA（VOLUME,SHORT）- MA（VOLUME,LONG））/MA（VOLUME,SHORT）×100
    """
    _c = pd.DataFrame()
    _c['trade_date'] = df['trade_date']
    _c['osc'] = (_ma(df.vol, n) - _ma(df.vol, m)) / _ma(df.vol, n) * 100
    return _c

def vol_per_change():
    vol =[0]
    for i in range(1,len(volume)):
        _vol_per_change = (volume[i]-volume[i-1])/volume[i] * 100
        vol.append(_vol_per_change)
    return pd.Series(vol)

def vhf(df, n=28):
    """
    纵横指标	vhf(28)
    VHF=（N日内最大收盘价与N日内最小收盘价之前的差）/（N日收盘价与前收盘价差的绝对值之和）
    """
    _vhf = pd.DataFrame()
    _vhf['trade_date'] = df.trade_date
    _vhf['vhf'] = (df.close.rolling(n).max() - df.close.rolling(n).min()) / (df.close - df.close.shift(1)).abs().rolling(n).sum()
    return _vhf

def asi(df, n=5):
    """
    振动升降指标(累计震动升降因子) ASI  # 同花顺给出的公式不完整就不贴出来了
    """
    _asi = pd.DataFrame()
    _asi['trade_date'] = df.trade_date
    _m = pd.DataFrame()
    _m['a'] = (df.high - df.close.shift()).abs()
    _m['b'] = (df.low - df.close.shift()).abs()
    _m['c'] = (df.high - df.low.shift()).abs()
    _m['d'] = (df.close.shift() - df.open.shift()).abs()
    _m['r'] = _m.apply(lambda x: x.a + 0.5 * x.b + 0.25 * x.d if max(x.a, x.b, x.c) == x.a else (
        x.b + 0.5 * x.a + 0.25 * x.d if max(x.a, x.b, x.c) == x.b else x.c + 0.25 * x.d
    ), axis=1)
    _m['x'] = df.close - df.close.shift() + 0.5 * (df.close - df.open) + df.close.shift() - df.open.shift()
    _m['k'] = np.maximum(_m.a, _m.b)
    _asi['si'] = 16 * (_m.x / _m.r) * _m.k
    _asi["asi"] = _ma(_asi.si, n)
    return _asi

def scaling(data):
    scaler = MinMaxScaler()
    scaled_data =[]
    if len(data)>250:
        for i in range(len(data)-250+1):
            if (i+250) != len(data):
                scaler.fit(data[i:(i+250)].to_numpy().reshape(-1, 1))
                scaled_data.append(scaler.transform(data[i].reshape(-1, 1)))
            else:
                scaler.fit(data[i:(i+250)].to_numpy().reshape(-1, 1))
                for a in range(250):
                    scaled_data.append(scaler.transform(data[a+i].reshape(-1, 1)))
    else:
        scaler.fit(data.to_numpy().reshape(-1, 1))
        for r in range(len(data)):
            scaled_data.append(scaler.transform(data[r].reshape(-1, 1)))
    final = np.array(scaled_data).flatten()
    return final
                    

def join_frame(d1, d2, column='trade_date'):
    return d1.join(d2.set_index(column), on=column)

def data_sequence(data, seq_len):    #### Data needs to be an array
    X = []
    y = []
    test_code =[]
    for i in range(seq_len,len(data)):
        truncated_data = data[i-seq_len:i]
        truncated_data = truncated_data.reset_index()
        y.append(truncated_data['Label'].to_list())
        del truncated_data['Label']
        X.append(truncated_data)
    test_code=data['ts_code'][0]
    return X, y, test_code


seq_len = 30

### Data prep
train_X = []
train_y = []
test_X = []
test_y = []
Stock=[]
directory = "D:/Trading_Bot/Data/Individual_Stock/"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        csv_file =os.path.join(directory, filename)
        original_data = pd.read_csv(csv_file)
        if len(original_data)>=150:
            closing_price = original_data['close']
            open_l = original_data['open']
            volume = original_data['vol']
            high = original_data['high']
            amount = original_data['amount']
            av_price = (original_data['amount']/original_data['vol'])*10/closing_price
            low = original_data['low']
            turn = original_data['turnover_rate']
            rsi_6 = ta.RSI(closing_price, timeperiod=6)
            rsi_12 = ta.RSI(closing_price, timeperiod=12)
            rsi_24 = ta.RSI(closing_price, timeperiod=24)
            original_data['rsi_6'] = scaling(rsi_6)
            original_data['rsi_12'] = scaling(rsi_12)
            original_data['rsi_12_dif'] = scaling(original_data['rsi_12'].diff())
            original_data['close_dif1'] = scaling(original_data['close'].pct_change(periods = 1))
            original_data['close_dif2'] = scaling(original_data['close'].pct_change(periods = 2))
            original_data['close_dif3'] = scaling(original_data['close'].pct_change(periods = 3))
            original_data['rsi_24'] = scaling(rsi_24)
            original_data['rsi_dif'] = scaling((rsi_6 - rsi_24))
            CJDX_J,CJDX_D = CJDX()
            original_data['J'] = scaling(CJDX_J)
            ##original_data['D'] = CJDX_D
            original_data['JD'] = scaling((CJDX_J - CJDX_D))
            rsi_dif = original_data['rsi_dif']
            MA_5 = ta.SMA(closing_price, timeperiod=5) / original_data['close']
            MA_10 = ta.SMA(closing_price, timeperiod=10) / original_data['close']
            MA_55 = ta.SMA(closing_price, timeperiod=55) / original_data['close']
            MA_144 = ta.SMA(closing_price, timeperiod=144) / original_data['close']
            original_data['MA_5'] = scaling(MA_5)
            original_data['MA_10'] = scaling(MA_10)
            original_data['MA_20'] = scaling(ta.SMA(closing_price, timeperiod=20) / original_data['close'])
            MA_20 = original_data['MA_20']
            MA_dif = MA_5 - MA_10
            original_data['MA_dif'] = scaling(MA_dif)
            SK_L, SD_L = CYHT()
            original_data['SD'] = scaling(SD_L)
            original_data['SK'] = scaling(SK_L)
            ##original_data['CYHT'] = SK_L - SD_L
            vol_MA = ta.SMA(volume, timeperiod=5) / original_data['vol']
            vol_MA20 = ta.SMA(volume, timeperiod=20) / original_data['vol']
            VMA_dif = vol_MA - vol_MA20
            original_data['VMA_dif'] = scaling(VMA_dif)
            vol_change_L = vol_per_change()
            original_data['vol_per'] = scaling(vol_change_L)
            original_data['vol_per2'] = scaling(volume.pct_change(periods = 2))
            vol_per=vol_change_L
            ATR = ta.ATR(high, low, closing_price, timeperiod=14)
            EMA_12 = ta.EMA(closing_price, timeperiod=12) / original_data['close']
            original_data['EMA_12'] = scaling(EMA_12)
            EMA_26 = ta.EMA(closing_price, timeperiod=26) / original_data['close']
            original_data['EMA_26'] = scaling(EMA_26)
            dif,dea,macd = MACD(closing_price)
            original_data['macd'] = scaling(macd)
            ##original_data['macd_dif'] = original_data['macd'].diff()
            ##original_data['MFI'] = ta.MFI(high, low, closing_price, volume, timeperiod=14)
            ##MFI =original_data['MFI']
            upperband, middleband, lowerband = ta.BBANDS(closing_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            original_data['BBupperband'] = scaling((upperband / closing_price))
            original_data['BBlowerband'] = scaling((lowerband / closing_price))
            lb = lowerband / closing_price
            ##asi_l = asi(original_data)
            ##original_data = join_frame(original_data, asi_l)
            ##vosc_l = vosc(original_data)
            ##original_data = join_frame(original_data, vosc_l)
            ##vhf_l = vhf(original_data)
            ##original_data = join_frame(original_data, vhf_l)
            WR_L = WRF()
            original_data['WR'] = scaling(WR_L)
            WR = WR_L
            MCST_L = MCST()
            original_data['MCST'] = scaling((MCST_L/closing_price))
            ##asil = original_data['asi']
            ##sil = original_data['si']
            ##oscl = original_data['osc']
            ##vhfl = original_data['vhf']
            ma5 = MA_5
            ma10 = MA_10
            ma55 = MA_55
            ma144 = MA_144
            ma20 = original_data['MA_20']
            PASR_L = pd.Series(PSAR(0.02))
            PSAR_S = ((PASR_L - closing_price)/closing_price)*100
            original_data['PSAR'] = scaling(PSAR_S)
            label_L = get_label()
            original_data['Label'] = label_L
            data_label_L = get_data_label()
            original_data['Data_Label'] = data_label_L
            actual_return_L = actual_return()
            original_data['Actual_Return'] = actual_return_L
            original_data['close'] = scaling(closing_price)
            original_data['open'] = scaling(open_l)
            original_data['vol'] = scaling(volume)
            original_data['high'] = scaling(high)
            original_data['amount'] = scaling(amount)
            original_data['low'] = scaling(low)
            original_data['turnover_rate'] = scaling(turn)
            original_data = original_data.loc[:, ~original_data.columns.str.contains('^Unnamed')]
            original_data = original_data.drop(columns=['pre_close','open','high','low','amount','trade_date'])
            original_data = original_data.drop(list(range(0, 145, 1)))
            X,y,test_code = data_sequence(original_data, seq_len)
            train_X.append(X[:-1])
            train_y.append(y[:-1])
            test_X.append(X[-1])
            test_y.append(y[-1])
            Stock.append(test_code)
            print(filename)
        else:
            continue



#data = pd.read_csv('D:\Trading_Bot\Data\prediction_All_data.csv')
#X = data.drop(['Label'], axis=1)
#y = data['Label']
#sm = SMOTE()
#X_res, y_res = sm.fit_sample(X, y.ravel())
#xgb_model = xgb.XGBClassifier(objective = "binary:logistic",tree_method= 'gpu_hist')
#xgb_model = xgb.XGBClassifier(objective = "binary:logistic",num_round = 1000)
#xgb_model = xgb.XGBClassifier()
#params = {
#            'eta': 0.30406,
#            'max_depth': 23,
#           'max_delta_step': 10,
#           'scale_pos_weight': 217.87,
#           'objective':  "binary:logistic"
#        }


#skf = StratifiedKFold(n_splits=10, shuffle = True)

#grid = GridSearchCV(xgb_model,
#                    param_grid = params,
#                    n_jobs = -1, 
#                    cv = skf.split(X_res, y_res)
#                    )



#grid.fit(X_res, y_res)

#best_pars = grid.best_params_
#param = params
#print(best_pars)
#dtrain = xgb.DMatrix(X_res,label=y_res)
#num_round = 100
#bst = xgb.Booster()
#bst = xgb.train(param, dtrain, num_round)
#model = bst
#bst.save_model('0001.model')
#bst.load_model('0001.model')
#result = {'Stock_Code':  [],'Accuracy': [],'Prediction':[],'Sector':[]}
#df = pd.DataFrame (result, columns = ['Stock_Code','Accuracy','Prediction','Sector'])
#print("Loaded model")
#prediction_data = pd.read_csv('../Data/prediction_All_data_test.csv')
#X_pred = prediction_data.drop(['Label','Stock_Code','Sector'], axis=1)
#y_pred = prediction_data['Label']
#dtest = xgb.DMatrix(X_pred)
##pred = model.predict(X_pred)
#pred = bst.predict(dtest)
#correct = 0
#for i in range(0, len(y_pred), 1):
#    if pred[i] == y_pred[i]:
#        correct = correct + 1
#accuracy = correct / len(y_pred)
#f['Prediction'] = pred
#f['Accuracy'] = accuracy
#df['Sector'] = prediction_data['Sector']
#outputpath="../Result/prediction_All.csv"
#df.to_csv(outputpath,sep=',',index=False,header=True)
