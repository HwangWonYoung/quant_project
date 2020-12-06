import requests
import traceback
import pandas as pd
import numpy as np
import time
from update.util.dbConnect import insert_data, exec_query
from datetime import datetime, timedelta
from itertools import groupby


def load_required_df():
    """ volatility 업데이트에 필요한 df 추출 """
    # 가장 최근 업데이트 날짜 추출
    weekly_volatility_latest_date = exec_query(f'select max(date) from stock_db.d_weekly_vol')
    idio_weekly_volatility_latest_date = exec_query(f'select max(date) from stock_db.d_idio_weekly_vol')
    if len(weekly_volatility_latest_date) != 0:
        weekly_volatility_latest_date = weekly_volatility_latest_date[0][0]
    else:
        weekly_volatility_latest_date = '20000101'

    if len(idio_weekly_volatility_latest_date) != 0:
        idio_weekly_volatility_latest_date = idio_weekly_volatility_latest_date[0][0]
    else:
        idio_weekly_volatility_latest_date = '20000101'

    # 가장 최근으로부터 2년 전 시점부터 데이터 추출
    from_date = str(
        datetime.strptime(weekly_volatility_latest_date, '%Y%m%d').date() - timedelta(days=365 * 2 + 30)).replace("-",
                                                                                                                  "")
    from_date2 = str(
        datetime.strptime(idio_weekly_volatility_latest_date, '%Y%m%d').date() - timedelta(days=365 * 2 + 30)).replace(
        "-", "")

    # 필요한 기간의 데이터 추출
    price = pd.DataFrame(
        exec_query(f'select `stock_cd`, `date`, `price` from stock_db.d_stock_price where date > {from_date}'))
    idio_rt = pd.DataFrame(
        exec_query(f'select `stock_cd`, `date`, `idio_rt` from stock_db.d_idio_rt where date > {from_date2}'))
    price.columns = ['stock_cd', 'date', 'price']
    idio_rt.columns = ['stock_cd', 'date', 'idio_rt']

    return price, idio_rt, weekly_volatility_latest_date, idio_weekly_volatility_latest_date


def custom_prod(df):
    return df.prod(skipna=False)


def cal_weekly_volatility(stock_rt, month, latest_date, idio=False):
    """ stock_rt이 주어졌을 때 지난 기간(month)의 weekly volatility를 return """

    from_index = int(np.where(stock_rt.index == latest_date)[0]) - 21 * month
    stock_rt = stock_rt.iloc[from_index:, ]

    stock_rt.index = pd.to_datetime(stock_rt.index, format='%Y%m%d')

    weekly_vol = []

    for i in range(len(stock_rt) - (21 * month)):
        stock_rt_part = 1 + stock_rt.iloc[i:(i + 21 * month + 1), :]
        stock_rt_part = (stock_rt_part.resample('W-MON').agg(custom_prod) - 1).std()
        weekly_vol += [stock_rt_part]

    weekly_vol = pd.concat(weekly_vol, 1).transpose()
    weekly_vol.index = stock_rt.tail(len(weekly_vol)).index

    weekly_vol = weekly_vol.reset_index().melt(id_vars="date", var_name="stock_cd", value_name="weekly_vol").dropna()
    weekly_vol = weekly_vol[['stock_cd', 'date', "weekly_vol"]]

    if idio:
        colname = "idio_weekly_vol_" + str(month) + "m"
    else:
        colname = "weekly_vol_" + str(month) + "m"

    weekly_vol = weekly_vol.rename(columns={'weekly_vol': colname})
    weekly_vol.date = weekly_vol.date.astype(str).str.replace("-", "")

    return weekly_vol


def return_weekly_volatility_set(price, idio_rt, volatility_latest_date, idio_volatility_latest_date):
    """ 계산 가능한 시기의 weekly volatility을 return """
    stock_rt = pd.pivot_table(price, values='price', index=['date'], columns=['stock_cd']).pct_change(1,
                                                                                                      fill_method=None).iloc[
               1:, ]
    idio_rt = pd.pivot_table(idio_rt, values='idio_rt', index=['date'], columns=['stock_cd'])

    weekly_vol_1m = cal_weekly_volatility(stock_rt, 1, volatility_latest_date)
    weekly_vol_3m = cal_weekly_volatility(stock_rt, 3, volatility_latest_date)
    weekly_vol_6m = cal_weekly_volatility(stock_rt, 6, volatility_latest_date)
    weekly_vol_12m = cal_weekly_volatility(stock_rt, 12, volatility_latest_date)
    weekly_vol_24m = cal_weekly_volatility(stock_rt, 24, volatility_latest_date)

    idio_weekly_vol_1m = cal_weekly_volatility(idio_rt, 1, idio_volatility_latest_date, idio=True)
    idio_weekly_vol_3m = cal_weekly_volatility(idio_rt, 3, idio_volatility_latest_date, idio=True)
    idio_weekly_vol_6m = cal_weekly_volatility(idio_rt, 6, idio_volatility_latest_date, idio=True)
    idio_weekly_vol_12m = cal_weekly_volatility(idio_rt, 12, idio_volatility_latest_date, idio=True)
    idio_weekly_vol_24m = cal_weekly_volatility(idio_rt, 24, idio_volatility_latest_date, idio=True)

    weekly_volatility_set = weekly_vol_1m.merge(
        weekly_vol_3m, how='left', on=['stock_cd', 'date']).merge(
        weekly_vol_6m, how='left', on=['stock_cd', 'date']).merge(
        weekly_vol_12m, how='left', on=['stock_cd', 'date']).merge(
        weekly_vol_24m, how='left', on=['stock_cd', 'date'])

    idio_weekly_volatility_set = idio_weekly_vol_1m.merge(
        idio_weekly_vol_3m, how='left', on=['stock_cd', 'date']).merge(
        idio_weekly_vol_6m, how='left', on=['stock_cd', 'date']).merge(
        idio_weekly_vol_12m, how='left', on=['stock_cd', 'date']).merge(
        idio_weekly_vol_24m, how='left', on=['stock_cd', 'date'])

    weekly_volatility_set = weekly_volatility_set.sort_values(by=['stock_cd', 'date'])
    idio_weekly_volatility_set = idio_weekly_volatility_set.sort_values(by=['stock_cd', 'date'])

    return weekly_volatility_set, idio_weekly_volatility_set


def update_weekly_vol_table():
    """ 서버db 업데이트(weekly_vol table & idio_weekly_vol table의 가장 최근 시점 이후의 volatility만을 기존 db에 append) """
    price, idio_rt, weekly_volatility_latest_date, idio_weekly_volatility_latest_date = load_required_df()

    updated_weekly_vol, updated_idio_weekly_vol = return_weekly_volatility_set(price, idio_rt,
                                                                               volatility_latest_date, idio_volatility_latest_date)
    updated_weekly_vol = updated_weekly_vol.loc[updated_weekly_vol.date > weekly_volatility_latest_date, :]
    updated_idio_weekly_vol = updated_idio_weekly_vol.loc[updated_idio_weekly_vol.date > idio_weekly_volatility_latest_date,:]

    return updated_weekly_vol, updated_idio_weekly_vol
