import requests
import traceback
import datetime
import pandas as pd
from bs4 import BeautifulSoup

KOSPI_old = pd.read_csv("data/KOSPI.csv")
KOSDAQ_old = pd.read_csv("data/KOSDAQ.csv")

url = "https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI"
res = requests.get(url)
res.encoding = 'utf-8'
res.status_code

soap = BeautifulSoup(res.text, 'lxml')

el_table_navi = soap.find("table", class_="Nnavi")
el_td_last = el_table_navi.find("td", class_="pgRR")
pg_last = el_td_last.a.get('href').rsplit('&')[1]
pg_last = pg_last.split('=')[1]
pg_last = int(pg_last)
pg_last

def parse_page(code, page):
    try:
        url = 'https://finance.naver.com/sise/sise_index_day.nhn?code={code}&page={page}'.format(code=code, page=page)
        res = requests.get(url)
        _soap = BeautifulSoup(res.text, 'lxml')
        _df = pd.read_html(str(_soap.find("table")), header=0)[0]
        _df = _df.dropna()
        return _df
    except Exception as e:
        traceback.print_exc()
    return None

str_datefrom = KOSPI_old.date[len(KOSPI_old.date)-1]
str_datefrom = str_datefrom.replace("-", ".")
str_dateto = datetime.datetime.strftime(datetime.datetime.today(), '%Y.%m.%d')

KOSPI_new = None
for page in range(1, pg_last+1):
    _df = parse_page("KOSPI", page)
    _df_filtered = _df[_df['날짜'] > str_datefrom]
    if KOSPI_new is None:
        KOSPI_new = _df_filtered
    else:
        KOSPI_new = pd.concat([KOSPI_new, _df_filtered])
    if len(_df) > len(_df_filtered):
        break

KOSDAQ_new = None
for page in range(1, pg_last+1):
    _df = parse_page("KOSDAQ", page)
    _df_filtered = _df[_df['날짜'] > str_datefrom]
    if KOSDAQ_new is None:
        KOSDAQ_new = _df_filtered
    else:
        KOSDAQ_new = pd.concat([KOSDAQ_new, _df_filtered])
    if len(_df) > len(_df_filtered):
        break

KOSPI_new = KOSPI_new[(['날짜', '체결가'])]
KOSPI_new.columns = ['date', 'close']
KOSPI_new = KOSPI_new.loc[::-1]
KOSPI_new.reset_index(drop=True, inplace=True)

KOSDAQ_new = KOSDAQ_new[(['날짜', '체결가'])]
KOSDAQ_new.columns = ['date', 'close']
KOSDAQ_new = KOSDAQ_new.loc[::-1]
KOSDAQ_new.reset_index(drop=True, inplace=True)

KOSPI_new.date = KOSPI_new.date.str.replace(".", "-")
KOSDAQ_new.date = KOSDAQ_new.date.str.replace(".", "-")

KOSPI_updated = pd.concat([KOSPI_old, KOSPI_new], axis=0).reset_index(drop=True)
KOSDAQ_updated = pd.concat([KOSDAQ_old, KOSDAQ_new], axis=0).reset_index(drop=True)

KOSPI_updated.to_csv("data/KOSPI.csv", index=False)
KOSDAQ_updated.to_csv("data/KOSDAQ.csv", index=False)