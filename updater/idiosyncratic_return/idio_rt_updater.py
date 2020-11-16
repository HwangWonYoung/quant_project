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
    """ idiosyncratic return 업데이트에 필요한 df 추출 """
    # 가장 최근 업데이트 날짜 추출
    idio_rt_latest_date = exec_query(f'select max(date) from stock_db.d_idio_rt')
    if len(idio_rt_latest_date) != 0:
        idio_rt_latest_date = idio_rt_latest_date[0][0]
    else:
        idio_rt_latest_date = '20000101'

    # 가장 최근으로부터 2년 전 시점부터 데이터 추출
    from_date = str(datetime.strptime(idio_rt_latest_date, '%Y%m%d').date() - timedelta(days=365 * 2 + 30)).replace("-", "")

    # 필요한 기간의 데이터 추출
    price = pd.DataFrame(
        exec_query(f'select `stock_cd`, `date`, `price` from stock_db.d_stock_price where date > {from_date}'))
    index = pd.DataFrame(exec_query(f'select* from stock_db.d_kospi_kosdaq where date > {from_date}'))
    market_info = pd.DataFrame(
        exec_query(f'select `date`, `stock_cd`, `market` from stock_db.stock_market_sector where date > {from_date}'))
    price.columns = ['stock_cd', 'date', 'price']
    index.columns = ['date', 'KOSPI', 'KOSDAQ']
    market_info.columns = ['date', 'stock_cd', 'market']

    # market_info의 date는 int로 되어있어서 chr로 변환 필요
    market_info['date'] = market_info['date'].astype('str')

    return price, index, market_info


def market_switched_codes(market_info):
    """ KOSPI, KOSDAQ 혹은 이전 상장한 종목코드 추출 """
    kospi_codes = set(market_info.loc[market_info.market == 'KOSPI', 'stock_cd'])
    kosdaq_codes = set(market_info.loc[market_info.market == 'KOSDAQ', 'stock_cd'])

    switched_codes = list(kospi_codes.intersection(kosdaq_codes))
    kospi_codes = list(kospi_codes - set(switched_codes))
    kosdaq_codes = list(kosdaq_codes - set(switched_codes))

    return switched_codes, kospi_codes, kosdaq_codes


def get_last_residual(market_rt, stock_rt):
    """ year년 동안의 시장수익률과 종목수익률로 회귀분석을 돌린 결과 마지막 시점의 잔차 (설명변수:시장수익률, 반응변수:종목수익률) """
    stock_rt_array, market_rt_array = np.array(stock_rt), np.array(market_rt)
    fitted_params = np.polyfit(market_rt_array, stock_rt_array, 1)
    predicted = np.polyval(fitted_params, market_rt_array)
    residuals = stock_rt_array - predicted
    return residuals[-1]


def cal_idiosyncratic_return(market_rt, stock_rt, year=2):
    """ n년간의 시장수익률과 종목수익률이 주어졌을 때 그 다음날의 고유수익을 return """
    idio_rt = []

    for i in range(len(market_rt) - (252 * year)):
        input_X = market_rt[i:(i + 252 * year + 1)]
        input_Y = stock_rt[i:(i + 252 * year + 1)]
        try:
            res = [get_last_residual(input_X, input_Y)]
        except:
            res = [None]
        idio_rt += res

    return idio_rt


def market_return_for_switched_stock(stock_cd, market_info, index):
    """ 이전상장된 종목코드가 주어졌을 때 이에 맞는 시장수익률을 return
         예를 들어, n년간 KOSDAQ에 상장되어 있다가 이후 KOSPI로 이전상장한 경우
         초기 n년은 KOSDAQ의 시장수익률, 그 이후는 KOSPI의 시장수익률이 들어감 """
    stock_cd_market_info = market_info.loc[market_info.stock_cd == stock_cd,]
    market_hist = list(stock_cd_market_info.market)
    market_change = [(k, sum(1 for i in g)) for k, g in groupby(market_hist)]

    market_rt = index.set_index('date').pct_change(1)

    if len(market_hist) < len(index):
        adjusted_index_return = [None] * (len(index) - len(market_hist))
    else:
        adjusted_index_return = []

    for i in range(len(market_change)):
        market_trace = market_change[i]
        trace_idx = market_trace[0]
        trace_len = market_trace[1]
        temp_index_return = market_rt[trace_idx][
                            len(adjusted_index_return):(len(adjusted_index_return) + trace_len)].tolist()
        adjusted_index_return += temp_index_return

    return adjusted_index_return[1:]


def return_idiosyncratic_set(price, index, market_info):
    """ 종목 데이터(개별 종목 가격 및 속한 시장)와 시장 데이터(KOSPI & KOSDAQ)이 주어졌을 때,
    계산 가능한 시기의 고유수익을 return """
    if ((set(price.date) == set(index.date) == set(market_info.date)) != True):
        stop("Dates should be equal")

    stock_rt = pd.pivot_table(price, values='price', index=['date'], columns=['stock_cd']).pct_change(1).iloc[1:, ]
    market_rt = index.set_index('date').pct_change(1).iloc[1:, ]

    switched_codes, kospi_codes, kosdaq_codes = market_switched_codes(market_info)
    kospi_stock_rt = stock_rt.loc[:, [i in kospi_codes for i in stock_rt.columns]]
    kosdaq_stock_rt = stock_rt.loc[:, [i in kosdaq_codes for i in stock_rt.columns]]
    switched_stock_rt = stock_rt.loc[:, [i in switched_codes for i in stock_rt.columns]]

    kospi_stock_idio_rt = kospi_stock_rt.apply(lambda x: cal_idiosyncratic_return(list(market_rt['KOSPI']), x), axis=0,
                                          result_type='expand')
    kosdaq_stock_idio_rt = kosdaq_stock_rt.apply(lambda x: cal_idiosyncratic_return(list(market_rt['KOSDAQ']), x), axis=0,
                                            result_type='expand')

    idio_rt_set = pd.concat([kospi_stock_idio_rt, kosdaq_stock_idio_rt], axis=1)

    switched_stock_idio_rt = dict()
    for i in switched_codes:
        adjusted_stock_rt = market_return_for_switched_stock(i, market_info, index)
        switched_stock_idio_rt[i] = cal_idiosyncratic_return(list(adjusted_stock_rt), list(switched_stock_rt[i]))
    switched_stock_idio_rt = pd.DataFrame(switched_stock_idio_rt)

    idio_rt_set = pd.concat([idio_rt_set, switched_stock_idio_rt], axis=1)

    idio_rt_set.index = stock_rt.tail(len(idio_rt_set)).index

    return idio_rt_set


def update_idio_rt_table():
    """ 서버db 업데이트(idio_rt table의 가장 최근 시점 이후의 idiosyncratic return만을 기존 db에 append) """
    idio_rt_latest_date = exec_query(f'select max(date) from stock_db.d_idio_rt')[0][0]

    price, index, market_info = load_required_df()

    if ((max(price.date) == max(index.date) == max(market_info.date)) != True):
        max_date = min(max(price.date), max(index.date), max(market_info.date))
        price = price.loc[price.date <= max_date, :]
        index = index.loc[index.date <= max_date, :]
        market_info = market_info.loc[market_info.date <= max_date, :]

    updated_idio_rt = return_idiosyncratic_set(price, index, market_info)
    updated_idio_rt = updated_idio_rt.reset_index().melt(id_vars="date", var_name="stock_cd", value_name="idio_rt")
    updated_idio_rt = updated_idio_rt[['stock_cd', 'date', 'idio_rt']]
    updated_idio_rt = updated_idio_rt.loc[updated_idio_rt.date > idio_rt_latest_date, :]

    return updated_idio_rt
