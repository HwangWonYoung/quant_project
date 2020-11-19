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
    """ beta 업데이트에 필요한 df 추출 """
    # 가장 최근 업데이트 날짜 추출
    beta_latest_date = exec_query(f'select max(date) from stock_db.d_beta')
    if len(beta_latest_date) != 0:
        beta_latest_date = beta_latest_date[0][0]
    else:
        beta_latest_date = '20000101'

    # 가장 최근으로부터 2년 전 시점부터 데이터 추출
    from_date = str(datetime.strptime(beta_latest_date, '%Y%m%d').date() - timedelta(days=365 * 2 + 30)).replace("-",
                                                                                                                 "")

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


def single_beta(market_rt, stock_rt):
    """ year년 동안의 시장수익률과 종목수익률로 회귀분석을 돌린 결과의 beta """
    stock_rt_array = np.array(stock_rt)
    market_rt_array = np.array(market_rt)
    fitted_params = np.polyfit(market_rt_array, stock_rt_array, 1)
    beta = fitted_params[0]

    return beta


def cal_beta(market_rt, stock_rt, year):
    """ n년간의 시장수익률과 종목수익률이 주어졌을 때 해당 기간의 beta를 return """
    beta_list = []

    for i in range(len(market_rt) - (252 * year)):
        input_X = market_rt[i:(i + 252 * year + 1)]
        input_Y = stock_rt[i:(i + 252 * year + 1)]
        try:
            beta = [single_beta(input_X, input_Y)]
        except:
            beta = [None]
        beta_list += beta

    return beta_list


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


def return_beta_set(price, index, market_info):
    """ 종목 데이터(개별 종목 가격 및 속한 시장)와 시장 데이터(KOSPI & KOSDAQ)이 주어졌을 때,
    계산 가능한 시기의 beta를 return """
    if ((set(price.date) == set(index.date) == set(market_info.date)) != True):
        stop("Dates should be equal")

    stock_rt = pd.pivot_table(price, values='price', index=['date'], columns=['stock_cd']).pct_change(1,
                                                                                                      fill_method=None).iloc[
               1:, ]
    market_rt = index.set_index('date').pct_change(1, fill_method=None).iloc[1:, ]

    switched_codes, kospi_codes, kosdaq_codes = market_switched_codes(market_info)
    kospi_stock_rt = stock_rt.loc[:, [i in kospi_codes for i in stock_rt.columns]]
    kosdaq_stock_rt = stock_rt.loc[:, [i in kosdaq_codes for i in stock_rt.columns]]
    switched_stock_rt = stock_rt.loc[:, [i in switched_codes for i in stock_rt.columns]]

    kospi_stock_beta_1y = kospi_stock_rt.apply(lambda x: cal_beta(list(market_rt['KOSPI']), x, year=1), axis=0,
                                               result_type='expand')
    kospi_stock_beta_2y = kospi_stock_rt.apply(lambda x: cal_beta(list(market_rt['KOSPI']), x, year=2), axis=0,
                                               result_type='expand')
    kosdaq_stock_beta_1y = kosdaq_stock_rt.apply(lambda x: cal_beta(list(market_rt['KOSDAQ']), x, year=1), axis=0,
                                                 result_type='expand')
    kosdaq_stock_beta_2y = kosdaq_stock_rt.apply(lambda x: cal_beta(list(market_rt['KOSDAQ']), x, year=2), axis=0,
                                                 result_type='expand')

    switched_stock_beta_1y = dict()
    switched_stock_beta_2y = dict()
    for i in switched_codes:
        adjusted_market_rt = market_return_for_switched_stock(i, market_info, index)
        switched_stock_beta_1y[i] = cal_beta(list(adjusted_market_rt), list(switched_stock_rt[i]), year=1)
        switched_stock_beta_2y[i] = cal_beta(list(adjusted_market_rt), list(switched_stock_rt[i]), year=2)

    switched_stock_beta_1y = pd.DataFrame(switched_stock_beta_1y)
    switched_stock_beta_2y = pd.DataFrame(switched_stock_beta_2y)

    beta_1y_set = pd.concat([kospi_stock_beta_1y, kosdaq_stock_beta_1y, switched_stock_beta_1y], axis=1)
    beta_2y_set = pd.concat([kospi_stock_beta_2y, kosdaq_stock_beta_2y, switched_stock_beta_2y], axis=1)

    beta_1y_set.index = stock_rt.tail(len(beta_1y_set)).index
    beta_2y_set.index = stock_rt.tail(len(beta_2y_set)).index

    return beta_1y_set, beta_2y_set


def update_beta_table():
    """ 서버db 업데이트(beta table의 가장 최근 시점 이후의 beta만을 기존 db에 append) """
    beta_latest_date = exec_query(f'select max(date) from stock_db.d_beta')[0][0]

    price, index, market_info = load_required_df()

    if ((max(price.date) == max(index.date) == max(market_info.date)) != True):
        max_date = min(max(price.date), max(index.date), max(market_info.date))
        price = price.loc[price.date <= max_date, :]
        index = index.loc[index.date <= max_date, :]
        market_info = market_info.loc[market_info.date <= max_date, :]

    updated_beta_1y, updated_beta_2y = return_beta_set(price, index, market_info)
    updated_beta_1y = updated_beta_1y.reset_index().melt(id_vars="date", var_name="stock_cd", value_name="beta_1y")
    updated_beta_2y = updated_beta_2y.reset_index().melt(id_vars="date", var_name="stock_cd", value_name="beta_2y")
    updated_beta = pd.merge(updated_beta_1y, updated_beta_2y, on=['date', 'stock_cd'], how='left').dropna()
    updated_beta = updated_beta[['stock_cd', 'date', 'beta_1y', 'beta_2y']]
    updated_beta = updated_beta.loc[updated_beta.date > beta_latest_date, :]

    return updated_beta