import pandas as pd
from pandas import DataFrame, Series
import requests as re
from bs4 import BeautifulSoup
import datetime as date
import time

USD_KRW_old = pd.read_csv("USD_KRW.csv")
JPY_KRW_old = pd.read_csv("JPY_KRW.csv")
EUR_KRW_old = pd.read_csv("EUR_KRW.csv")
CNY_KRW_old = pd.read_csv("CNY_KRW.csv")

last_update_date = USD_KRW_old.date[len(USD_KRW_old.date)-1]


def market_index_crawling(pages, last_update_date):
    final_dict = {}

    url_dict = {'USD_KRW': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_USDKRW',
                'JPY_KRW': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_JPYKRW',
                'EUR_KRW': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_EURKRW',
                'CNY_KRW': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_CNYKRW'}

    for key in url_dict.keys():

        date = []
        value = []

        for i in range(1, pages):

            url = re.get(url_dict[key] + '&page=%s' % i)
            url = url.content

            html = BeautifulSoup(url, 'html.parser')

            tbody = html.find('tbody')
            tr = tbody.find_all('tr')

            for r in tr:
                temp_date = r.find('td', {'class': 'date'}).text.replace('.', '-').strip()
                temp_value = r.find('td', {'class': 'num'}).text.strip()

                date.append(temp_date)
                value.append(temp_value)

        final_dict[key] = {'date': date, 'close': value}

    USD_KRW = pd.DataFrame(final_dict['USD_KRW']).loc[::-1].reset_index(drop=True)
    JPY_KRW = pd.DataFrame(final_dict['JPY_KRW']).loc[::-1].reset_index(drop=True)
    EUR_KRW = pd.DataFrame(final_dict['EUR_KRW']).loc[::-1].reset_index(drop=True)
    CNY_KRW = pd.DataFrame(final_dict['CNY_KRW']).loc[::-1].reset_index(drop=True)

    USD_KRW.close = USD_KRW.close.str.replace(",", "")
    JPY_KRW.close = JPY_KRW.close.str.replace(",", "")
    EUR_KRW.close = EUR_KRW.close.str.replace(",", "")
    CNY_KRW.close = CNY_KRW.close.str.replace(",", "")

    USD_KRW = USD_KRW.loc[USD_KRW.date > last_update_date, :]
    JPY_KRW = JPY_KRW.loc[JPY_KRW.date > last_update_date, :]
    EUR_KRW = EUR_KRW.loc[EUR_KRW.date > last_update_date, :]
    CNY_KRW = CNY_KRW.loc[CNY_KRW.date > last_update_date, :]

    return USD_KRW, JPY_KRW, EUR_KRW, CNY_KRW

USD_KRW_new, JPY_KRW_new, EUR_KRW_new, CNY_KRW_new = market_index_crawling(30, last_update_date)

USD_KRW_updated = pd.concat([USD_KRW_old, USD_KRW_new], axis=0).reset_index(drop=True)
JPY_KRW_updated = pd.concat([JPY_KRW_old, JPY_KRW_new], axis=0).reset_index(drop=True)
EUR_KRW_updated = pd.concat([EUR_KRW_old, EUR_KRW_new], axis=0).reset_index(drop=True)
CNY_KRW_updated = pd.concat([CNY_KRW_old, CNY_KRW_new], axis=0).reset_index(drop=True)

USD_KRW_updated.to_csv("USD_KRW.csv", index=False)
JPY_KRW_updated.to_csv("JPY_KRW.csv", index=False)
EUR_KRW_updated.to_csv("EUR_KRW.csv", index=False)
CNY_KRW_updated.to_csv("CNY_KRW.csv", index=False)