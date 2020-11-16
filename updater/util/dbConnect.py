
from pymysql import connect
from sqlalchemy import create_engine
from datetime import datetime


def exec_query(sql):
    # MySQL Connection 연결
    # conn = connect(host='stockbalance.duckdns.org', port=10005, user='shiraz', password='stockbalance124816',
    #                db='sauvignon', charset='utf8')
    conn = connect(host='betterlife.duckdns.org', port=1231, user='betterlife', password='snail132',
                   db='stock_db', charset='utf8')

    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()

    # SQL문 실행
    sql = sql
    curs.execute(sql)

    # 데이타 Fetch
    result = curs.fetchall()

    # Connection 닫기
    conn.commit()
    conn.close()

    return result


def insert_data(data, table_name):
    # db connection get
    engine = create_engine(
        # "mysql+pymysql://shiraz:stockbalance124816@stockbalance.duckdns.org:10005/sauvignon?charset=utf8",
        "mysql+pymysql://betterlife:snail132@betterlife.duckdns.org:1231/stock_db?charset=utf8",
        encoding='utf-8')

    # 임시 테이블에 저장
    tmp_table = f'tmp_table_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    data.to_sql(tmp_table, con=engine, if_exists='append', index=False)

    # target 테이블에 임시 테이블 내용 insert(duplicated key error 발생 시, replace함)
    exec_query(f'replace into {table_name} select * from {tmp_table}')

    # 임시테이블 삭제
    exec_query(f'drop table {tmp_table}')

    # 연결 종료
    engine.dispose()
