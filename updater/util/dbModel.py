
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# sqlalchemy 설정
engine = create_engine('mysql+pymysql://betterlife:snail132@betterlife.duckdns.org:1231/stock_db')
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()


def get_connection():
    return engine.connect()


def get_session():
    return db_session


class StockIsinStockCdMap(Base):
    __tablename__ = 'stock_isin_stock_cd_map'

    isin_cd = Column(String, primary_key=True)
    stock_cd = Column(String, primary_key=True)

    def __init__(self, isin_cd, stock_cd):
        self.isin_cd = isin_cd
        self.stock_cd = stock_cd
