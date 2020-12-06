import requests
import traceback
import pandas as pd
import numpy as np
import time
from update.util.dbConnect import insert_data, exec_query
from datetime import datetime, timedelta
from itertools import groupby


def load_required_df():
    """ daily volatility 업데이트에 필요한 df 추출 """
    # 가장 최근 업데이트 날짜 추출
    daily_volatility_latest_date = exec_query(f'select max(date) from stock_db.d_daily_vol')
    idio_daily_volatility_latest_date = exec_query(f'select max(date) from stock_db.d_idio_daily_vol')
    if len(daily_volatility_latest_date) != 0:
        daily_volatility_latest_date = daily_volatility_latest_date[0][0]
    else:
        daily_volatility_latest_date = '20000101'

    if len(idio_daily_volatility_latest_date) != 0:
        idio_daily_volatility_latest_date = idio_daily_volatility_latest_date[0][0]
    else:
        idio_daily_volatility_latest_date = '20000101'

    # 가장 최근으로부터 2년 전 시점부터 데이터 추출
    from_date = str(
        datetime.strptime(daily_volatility_latest_date, '%Y%m%d').date() - timedelta(days=365 * 2 + 30)).replace("-",
                                                                                                                 "")
    from_date2 = str(
        datetime.strptime(idio_daily_volatility_latest_date, '%Y%m%d').date() - timedelta(days=365 * 2 + 30)).replace(
        "-", "")

    # 필요한 기간의 데이터 추출
    price = pd.DataFrame(
        exec_query(f'select `stock_cd`, `date`, `price` from stock_db.d_stock_price where date > {from_date}'))
    idio_rt = pd.DataFrame(
        exec_query(f'select `stock_cd`, `date`, `idio_rt` from stock_db.d_idio_rt where date > {from_date2}'))
    price.columns = ['stock_cd', 'date', 'price']
    idio_rt.columns = ['stock_cd', 'date', 'idio_rt']

    return price, idio_rt, daily_volatility_latest_date, idio_daily_volatility_latest_date


def cal_daily_volatility(stock_rt, month, latest_date, idio=False):
    """ stock_rt이 주어졌을 때 지난 기간(month)의 daily volatility를 return """

    from_index = int(np.where(stock_rt.index == latest_date)[0]) - 21 * month
    stock_rt = stock_rt.iloc[from_index:, ]

    vol_wide = stock_rt.rolling(window=1 + 21 * month).std()
    vol_long = vol_wide.reset_index().melt(id_vars="date", var_name="stock_cd", value_name="daily_vol").dropna()
    vol_long = vol_long[['stock_cd', 'date', "daily_vol"]]

    if idio:
        colname = "idio_daily_vol_" + str(month) + "m"
    else:
        colname = "daily_vol_" + str(month) + "m"

    vol_long = vol_long.rename(columns={'daily_vol': colname})

    return vol_long


def return_daily_volatility_set(price, idio_rt, volatility_latest_date, idio_volatility_latest_date):
    """ 계산 가능한 시기의 daily volatility을 return """

    stock_rt = pd.pivot_table(price, values='price', index=['date'], columns=['stock_cd']).pct_change(1,
                                                                                                      fill_method=None).iloc[
               1:, ]
    idio_rt = pd.pivot_table(idio_rt, values='idio_rt', index=['date'], columns=['stock_cd'])

    daily_vol_1m = cal_daily_volatility(stock_rt, 1, volatility_latest_date)
    daily_vol_3m = cal_daily_volatility(stock_rt, 3, volatility_latest_date)
    daily_vol_6m = cal_daily_volatility(stock_rt, 6, volatility_latest_date)
    daily_vol_12m = cal_daily_volatility(stock_rt, 12, volatility_latest_date)
    daily_vol_24m = cal_daily_volatility(stock_rt, 24, volatility_latest_date)

    idio_daily_vol_1m = cal_daily_volatility(idio_rt, 1, idio_volatility_latest_date, idio=True)
    idio_daily_vol_3m = cal_daily_volatility(idio_rt, 3, idio_volatility_latest_date, idio=True)
    idio_daily_vol_6m = cal_daily_volatility(idio_rt, 6, idio_volatility_latest_date, idio=True)
    idio_daily_vol_12m = cal_daily_volatility(idio_rt, 12, idio_volatility_latest_date, idio=True)
    idio_daily_vol_24m = cal_daily_volatility(idio_rt, 24, idio_volatility_latest_date, idio=True)

    daily_volatility_set = daily_vol_1m.merge(
        daily_vol_3m, how='left', on=['stock_cd', 'date']).merge(
        daily_vol_6m, how='left', on=['stock_cd', 'date']).merge(
        daily_vol_12m, how='left', on=['stock_cd', 'date']).merge(
        daily_vol_24m, how='left', on=['stock_cd', 'date'])

    idio_daily_volatility_set = idio_daily_vol_1m.merge(
        idio_daily_vol_3m, how='left', on=['stock_cd', 'date']).merge(
        idio_daily_vol_6m, how='left', on=['stock_cd', 'date']).merge(
        idio_daily_vol_12m, how='left', on=['stock_cd', 'date']).merge(
        idio_daily_vol_24m, how='left', on=['stock_cd', 'date'])

    daily_volatility_set = daily_volatility_set.sort_values(by=['stock_cd', 'date'])
    idio_daily_volatility_set = idio_daily_volatility_set.sort_values(by=['stock_cd', 'date'])

    return daily_volatility_set, idio_daily_volatility_set


def update_daily_vol_table():
    """ 서버db 업데이트(daily_vol table & idio_daily_vol table의 가장 최근 시점 이후의 volatility만을 기존 db에 append) """
    price, idio_rt, daily_volatility_latest_date, idio_daily_volatility_latest_date = load_required_df()

    updated_daily_vol, updated_idio_daily_vol = return_daily_volatility_set(price, idio_rt,
                                                                            daily_volatility_latest_date,
                                                                            idio_daily_volatility_latest_date)
    updated_daily_vol = updated_daily_vol.loc[updated_daily_vol.date > daily_volatility_latest_date, :]
    updated_idio_daily_vol = updated_idio_daily_vol.loc[updated_idio_daily_vol.date > idio_daily_volatility_latest_date,
                             :]

    return updated_daily_vol, updated_idio_daily_vol