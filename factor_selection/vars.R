# will be updated when NEW factor added

pos_vars <- c("ROA_1", "ROA_2", "ROE_1", "ROE_2", "CFO", "ACCURUAL_1", "ACCURUAL_2",
              "LIQUIDITY", "MARGIN", "OP_MARGIN", "TURNOVER", "GPA", "ROC", "ROIC",
              "sales_growth_QoQ", "op_growth_QoQ", "net_growth_QoQ", "sales_growth_YoY",
              "op_growth_YoY", "net_growth_YoY", "sales_growth_3YoY", "op_growth_3YoY",
              "net_growth_3YoY", "op_turn_profit", "net_turn_profit", "EY_r", "OEY_r",
              "CFY_r", "idiosyncratic_momentum_6_2", "idiosyncratic_momentum_12_2",
              "momentum_6_2", "momentum_12_2", "volume_momentum_1", "volume_momentum_3",
              "volume_momentum_6", "beta_one_year", "beta_two_year")

neg_vars <- c("LEVERAGE", "DE", "op_turn_loss", "net_turn_loss", "PBR_r", "PSR_r",
              "idiosyncratic_daily_vol_oneyear", "idiosyncratic_weekly_vol_oneyear",
              "daily_vol_oneyear", "daily_vol_twoyear", "weekly_vol_oneyear", "weekly_vol_twoyear",
              "momentum_1", "MDD")

Y_vars <- c("Y_2_week", "Y_1_month", "Y_2_month", "Y_3_month", "Y_6_month", "Y_12_month",
            "sharpe_ratio_1m_Y", "sharpe_ratio_3m_Y", "sharpe_ratio_6m_Y")

etc_vars <- c("name", "code", "market", "sector", "price", "market_cap", "size_level") # do not use size factor