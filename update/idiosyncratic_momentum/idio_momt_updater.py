import requests
import traceback
import pandas as pd
import numpy as np
import time
from update.util.dbConnect import insert_data, exec_query
from datetime import datetime, timedelta
from itertools import groupby


def load_required_df():
    """ idiosyncratic momentum 업데이트에 필요한 df 추출 """
    # 가장 최근 업데이트 날짜 추출
    idio_momt_latest_date = exec_query(f'select max(date) from stock_db.d_idio_momt')
    if len(idio_momt_latest_date) != 0:
        idio_momt_latest_date = idio_momt_latest_date[0][0]
    else:
        idio_momt_latest_date = '20000101'

    # 가장 최근으로부터 2년 전 시점부터 데이터 추출
    from_date = str(datetime.strptime(idio_momt_latest_date, '%Y%m%d').date() - timedelta(days=365 * 2 + 30)).replace(
        "-", "")

    # 필요한 기간의 데이터 추출
    idio_rt = pd.DataFrame(
        exec_query(f'select `stock_cd`, `date`, `idio_rt` from stock_db.d_idio_rt where date > {from_date}'))
    idio_rt.columns = ['stock_cd', 'date', 'idio_rt']

    return idio_rt, idio_momt_latest_date


def cal_idio_momt(idio_rt, month, latest_date, ignore_last_month=False):
    """ idio_rt이 주어졌을 때 지난 기간(month)의 idiosyncratic momentum을 return """

    from_index = int(np.where(idio_rt.index == latest_date)[0]) - 21 * month
    idio_rt = idio_rt.iloc[from_index:, ]

    idio_momt = []

    if ignore_last_month:
        for i in range(len(idio_rt) - (21 * month)):
            idio_rt_part = 1 + idio_rt.iloc[i:(i + 21 * month + 1), :]
            idio_rt_part = idio_rt_part.head(21 * month - 20)
            idio_momt_part = idio_rt_part.prod(skipna=False) - 1
            idio_momt += [idio_momt_part]
    else:
        for i in range(len(idio_rt) - (21 * month)):
            idio_rt_part = 1 + idio_rt.iloc[i:(i + 21 * month + 1), :]
            idio_momt_part = idio_rt_part.prod(skipna=False) - 1
            idio_momt += [idio_momt_part]

    idio_momt = pd.concat(idio_momt, 1).transpose()
    idio_momt.index = idio_rt.tail(len(idio_momt)).index

    idio_momt = idio_momt.reset_index().melt(id_vars="date", var_name="stock_cd", value_name="idio_momt").dropna()
    idio_momt = idio_momt[['stock_cd', 'date', "idio_momt"]]

    if ignore_last_month:
        colname = "idio_momt_2m_" + str(month) + "m"
    else:
        colname = "idio_momt_" + str(month) + "m"

    idio_momt = idio_momt.rename(columns={'idio_momt': colname})

    return idio_momt


def return_idio_momt_set(idio_rt, idio_momt_latest_date):
    """ 계산 가능한 시기의 idiosyncratic momentum을 return """

    idio_rt = pd.pivot_table(idio_rt, values='idio_rt', index=['date'], columns=['stock_cd'])

    idio_momt_1m = cal_idio_momt(idio_rt_wide, 1, idio_momt_latest_date, ignore_last_month=False)
    idio_momt_3m = cal_idio_momt(idio_rt_wide, 3, idio_momt_latest_date, ignore_last_month=False)
    idio_momt_2m_6m = cal_idio_momt(idio_rt_wide, 6, idio_momt_latest_date, ignore_last_month=True)
    idio_momt_2m_12m = cal_idio_momt(idio_rt_wide, 12, idio_momt_latest_date, ignore_last_month=True)
    idio_momt_2m_24m = cal_idio_momt(idio_rt_wide, 24, idio_momt_latest_date, ignore_last_month=True)

    idio_momt_set = idio_momt_1m.merge(
        idio_momt_3m, how='left', on=['stock_cd', 'date']).merge(
        idio_momt_2m_6m, how='left', on=['stock_cd', 'date']).merge(
        idio_momt_2m_12m, how='left', on=['stock_cd', 'date']).merge(
        idio_momt_2m_24m, how='left', on=['stock_cd', 'date'])

    idio_momt_set = idio_momt_set.sort_values(by=['stock_cd', 'date'])

    return idio_momt_set


def update_idio_momt_table():
    """ 서버db 업데이트(idio momt table의 가장 최근 시점 이후의 idiosyncratic momentum만을 기존 db에 append) """
    idio_momt_latest_date = exec_query(f'select max(date) from stock_db.d_idio_momt')[0][0]
    idio_rt, idio_momt_latest_date = load_required_df()

    updated_idio_momt = return_idio_momt_set(idio_rt, idio_momt_latest_date)
    updated_idio_momt = updated_idio_momt.loc[updated_idio_momt.date > idio_momt_latest_date, :]

    return updated_idio_momt