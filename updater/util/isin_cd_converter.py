
from update.util.dbModel import get_session, StockIsinStockCdMap
import requests
from bs4 import BeautifulSoup


def isin_to_stock_cd(isin_cd):
    db_session = get_session()
    isin_map = db_session.query(StockIsinStockCdMap).filter(StockIsinStockCdMap.isin_cd == isin_cd).first()

    if isin_map is None:
        # isin_cd 크롤링
        isin_url = 'https://isin.krx.co.kr/srch/srch.do?method=srchPopup2'
        isin_data = {
            'stdcd_type': '2',
            'std_cd': isin_cd,
        }

        r = requests.post(isin_url, data=isin_data)
        soup = BeautifulSoup(r.content, 'lxml')
        stock_cd = soup.select('#wrapper-pop > div > table > tbody > tr:nth-child(2) > td.last')[0].text

        if len(stock_cd) > 6:
            stock_cd = ''

        # DB 추가
        print("isin_cd mapping table update - isin_cd: "+isin_cd+" stock_cd: "+stock_cd)
        db_session.add(StockIsinStockCdMap(isin_cd=isin_cd, stock_cd=stock_cd))
        db_session.commit()

        return stock_cd
    else:
        return isin_map.stock_cd


if __name__ == '__main__':
    print(isin_to_stock_cd("KR7005930003"))
