import requests
import traceback
import pandas as pd
import numpy as np
import time
import os
from update.util.dbConnect import insert_data, exec_query
from datetime import datetime, timedelta
from itertools import groupby


def load_required_df():
    """ mdd 업데이트에 필요한 df 추출 """
    # 가장 최근 업데이트 날짜 추출
    mdd_latest_date = exec_query(f'select max(date) from stock_db.d_mdd')
    if len(mdd_latest_date) != 0:
        mdd_latest_date = mdd_latest_date[0][0]
    else:
        mdd_latest_date = '20000101'

    # 가장 최근으로부터 2년 전 시점부터 데이터 추출
    from_date = str(datetime.strptime(mdd_latest_date, '%Y%m%d').date() - timedelta(days=365 * 2 + 30)).replace("-", "")

    # 필요한 기간의 데이터 추출
    price = pd.DataFrame(
        exec_query(f'select `stock_cd`, `date`, `price` from stock_db.d_stock_price where date > {from_date}'))
    price.columns = ['stock_cd', 'date', 'price']

    return price


def cal_mdd(stock_pr, month):
    """ n달 간의 종목가격이 주어졌을 때 그 기간의 mdd를 return """
    mdd = []

    for i in range(len(stock_pr) - (21 * month)):
        price_part = stock_pr[i:(i + 21 * month + 1)]
        try:
            max_price = max(price_part)
            min_price = min(price_part)
            mdd_part = (min_price - max_price) / max_price
        except:
            mdd_part = [None]
        mdd += [mdd_part]

    return mdd


def cal_mdd2(price_wide, month):
    """ cal_mdd를 column wise 하게 df에 적용하여 tidy form의 결과물을 return """

    mdd_wide = price_wide.apply(lambda x: cal_mdd(x, month), axis=0, result_type='expand')
    mdd_wide.index = price_wide.tail(len(mdd_wide)).index
    mdd_long = mdd_wide.reset_index().melt(id_vars="date", var_name="stock_cd", value_name="mdd").dropna()
    mdd_long = mdd_long[['stock_cd', 'date', "mdd"]]
    colname = "mdd_" + str(month) + "m"
    mdd_long = mdd_long.rename(columns={'mdd': colname})

    return mdd_long


def return_mdd_set(price):
    """ 계산 가능한 시기의 mdd를 return """

    price_wide = pd.pivot_table(price, values='price', index=['date'], columns=['stock_cd'])

    mdd_1m = cal_mdd2(price_wide, 1)
    mdd_3m = cal_mdd2(price_wide, 3)
    mdd_6m = cal_mdd2(price_wide, 6)
    mdd_12m = cal_mdd2(price_wide, 12)
    mdd_24m = cal_mdd2(price_wide, 24)

    mdd_set = mdd_1m.merge(
        mdd_3m, how='left', on=['stock_cd', 'date']).merge(
        mdd_6m, how='left', on=['stock_cd', 'date']).merge(
        mdd_12m, how='left', on=['stock_cd', 'date']).merge(
        mdd_24m, how='left', on=['stock_cd', 'date'])

    mdd_set = mdd_set.sort_values(by=['stock_cd', 'date'])

    return mdd_set


def update_mdd_table():
    """ 서버db 업데이트(mdd table의 가장 최근 시점 이후의 mdd만을 기존 db에 append) """
    mdd_latest_date = exec_query(f'select max(date) from stock_db.d_mdd')[0][0]

    price = load_required_df()

    updated_mdd = return_mdd_set(price)
    updated_mdd = updated_mdd.loc[updated_mdd.date > mdd_latest_date, :]

    return updated_mdd