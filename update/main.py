
from update.krx.krxMarketCapDataUpdater import tb_update_stock_daily_technical
from update.krx.krxMarketSectorDataUpdater import update_stock_market_sector_table
from update.naver.kospi_kosdaq_updater import tb_update_kospi_kosdaq


if __name__ == '__main__':
    tb_update_stock_daily_technical()
    update_stock_market_sector_table()
    tb_update_kospi_kosdaq()
