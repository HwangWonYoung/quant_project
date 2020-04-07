# import libraries
library("data.table")
library("dplyr")
library("ggplot2")
library("tibble")
library("lubridate")
library("xts")
library("stringr")
library("xts")
library("PerformanceAnalytics")
library("quantmod")
library("dplyr")
library("purrr")
library("ggplot2")
library("magrittr")
library("lazyeval")

# path
modeling_path <- "C:/Users/82106/Documents/quant_project/backtesting/modeling_sets/"
econ_distance_path <- "C:/Users/82106/Documents/quant_project/backtesting/econ_distance/"

# macro index
KOSPI <- read.csv(paste0(econ_distance_path, "KOSPI_match_result_120.csv"), row.names = 1)
KOSDAQ <- read.csv(paste0(econ_distance_path, "KOSDAQ_match_result_120.csv"), row.names = 1)
KOR1 <- read.csv(paste0(econ_distance_path, "kor_1_year_match_result_120.csv"), row.names = 1) # 1년 국채 금리
KOR3 <- read.csv(paste0(econ_distance_path, "kor_3_year_match_result_120.csv"), row.names = 1) # 3년 국채 금리, 국채와 금리의 관계 다시 한 번 공부해보자 
USA1 <- read.csv(paste0(econ_distance_path, "usa_1_year_match_result_120.csv"), row.names = 1) # 1년 미국 국채 금리
USA3 <- read.csv(paste0(econ_distance_path, "usa_3_year_match_result_120.csv"), row.names = 1) # 3년 미국 국채 금리 
EUR <- read.csv(paste0(econ_distance_path, "eur_match_result_120.csv"), row.names = 1)
JPY <- read.csv(paste0(econ_distance_path, "jpy_match_result_120.csv"), row.names = 1)
USD <- read.csv(paste0(econ_distance_path, "usd_match_result_120.csv"), row.names = 1)
WTI <- read.csv(paste0(econ_distance_path, "wti_match_result_120.csv"), row.names = 1)

# factors
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

# history of stock price
KOR_price <- read.csv(paste0("C:/Users/82106/Documents/quant_project/backtesting/KOR_price_20200203.csv"), row.names = 1, stringsAsFactors = FALSE, fileEncoding = "CP949", encoding = "UTF-8")
colnames(KOR_price) <- colnames(KOR_price) %>% str_remove_all("X")

# get first buiseness day of each month  
Date <- rownames(KOR_price)
Date <- ymd(Date)
Date <- as.data.frame(Date) ; colnames(Date) <- "date"
first_bizday_of_month <-
  Date %>% 
  mutate(year = date %>% year, month = date %>% month) %>% 
  group_by(year, month) %>% 
  slice(which.min(date)) %>% data.frame %>%
  select(date) 

first_bizday_of_month <- first_bizday_of_month$date
rm(Date) 

# function setting date window
set_date_window <- function(date_vec, start_date, end_date){
  window <-date_vec[date_vec >= start_date & date_vec <= end_date]
  return(window)
}

# modeling_date : 모델링에 사용되는 데이터 기간
# testing_date : 테스트에 사용되는 데이터 기간 
modeling_date <- set_date_window(first_bizday_of_month, "2014-04-01", "2019-12-02")
testing_date <- set_date_window(first_bizday_of_month, "2018-01-02", "2019-12-02")

# 유사도를 고려할 때 선택해야 하는 것들은
# 어떤 macro 지표를 사용할 것인가, 가장 가까운 국면 몇 개를 선택할 것인가
# 사용할 macro를 선택하여 하나의 macro distance set으로 만든다
MACRO <- KOSPI + KOSDAQ  # + EUR + JPY + USD + KOR1 + KOR3 + USA1 + USA3 + WTI

# functions used in back test code
# test 데이터와 유사 시점이라고 판단되는 data set에 적용되는 pre process 함수
# 값이 클 수록 좋은 factor, 작을 수록 좋은 factor에 맞게 z-scoring 진행
preprocess_for_factor_scoring <- function(df, pos_vars, neg_vars, Y){
  
  X <- colnames(df)[which(colnames(df) %in% c(pos_vars, neg_vars))]
  df <- df %>% select(Y, X)
  
  # NA 비율이 20%가 넘는 column 삭제, complete obs만 사용 
  df <- df %>% select(colnames(df)[sapply(df, function(x){sum(is.na(x))/length(x)}) <= 0.2])
  df <- df[complete.cases(df),]
  
  Y_col <- colnames(df)[1]
  X_cols <- colnames(df)[-1]
  
  # Z-scoring
  for(i in X_cols){
    if(i %in% pos_vars){
      # 클 수록 좋은 팩터  
      df[i] <- scale(min_rank(desc(df[i]))) %>% as.vector() # scaling 후 matrix format으로 변하는 것 방지 
    }else{
      # 작을 수록 좋은 팩터  
      df[i] <- scale(min_rank(df[i])) %>% as.vector()
    }
  }
  
  return(df)
  
}

# pre process된 팩터 데이터 내에서 주어진 Y(i.e. Y_1_month) 에 대해 각 팩터 포트폴리오가 얼마만큼의 Return을 냈는지 확인 
# 포트폴리오를 몇 개로 구성하는지는 선택할 수 있음, default는 30개 종목  
detect_outperform_factors <- function(df, stock_numbers = 30){
  
  Y_col <- colnames(df)[1]
  X_cols <- colnames(df)[-1]
  
  output <- list()
  
  for(i in X_cols){
    
    tmp_df <- df %>% select(Y_col, i)
    tmp_df <- tmp_df %>% arrange_(interp(~desc(var), var = as.name(i))) %>% head(stock_numbers)
    Rt <- mean(tmp_df[[Y_col]])
    output[[i]] <- Rt
    
  }
  
  output <- output %>% as.data.frame() %>% t() ; colnames(output) <- "Rt" ; output <- as.data.frame(output)
  output <- rownames_to_column(output, "factor")
  output <- arrange(output, desc(Rt))
  
  return(output)
  
}

# test set 에 대해서 주어진 팩터로 선정된 종목을 산출하는 함수
stocks_for_portfolio <- function(df){
  
  # 정보가 complete한 종목만 뽑기
  df <- df[complete.cases(df),]
  
  code <- df$code
  factor_scores <- df %>% select(-code) %>% rowSums()
  output <- data.frame(code, factor_scores) ; colnames(output) <- c("code", "factor_score")
  output <- output %>% arrange(desc(factor_scores)) %>% head(30)
  
  return(output$code)
}

ReturnsWithSelectedStocks <- list()

for(i in 1:length(testing_date)){
  
  # test data와 매칭되는 날짜의 macro index 거리 추출 
  macro_dist <- MACRO %>% slice(which(ymd(rownames(MACRO)) %in% testing_date[i])) %>% t() %>% as.data.frame()
  
  # 유사도가 측정되는 시점은 최소 test data의 날짜와 6개월 차이가 남
  # 그 이유는 패턴 비교의 기준이 되는 window와는 겹치는 부분을 없앴기 때문 
  macro_dist$date <- first_bizday_of_month[(which(first_bizday_of_month %in% testing_date[i])-35) : 
                                             (which(first_bizday_of_month %in% testing_date[i])-6)]
  
  colnames(macro_dist)[1] <- "distance"
  
  # 가장 가까운 n개의 시점 추출, 여기선 10개 
  closest_date <- arrange(macro_dist, distance) %>% head(10) %>% select(date)
  
  # 날짜가 추출되었다면, 그 시점에서 아웃퍼폼한 팩터들을 확인한다
  closest_data <- data.frame()
  
  for(j in 1:10){
    tmp <- read.csv(paste0(modeling_path,closest_date$date[j],"-data.csv")) %>% select(-etc_vars)
    closest_data <- dplyr::bind_rows(closest_data, tmp)
  }
  
  closest_data <- preprocess_for_factor_scoring(closest_data, pos_vars, neg_vars, Y_vars[2])
  
  # 가까운 시점에서 아웃퍼폼한 팩터 30개를 추출, 여기서는 30개 
  selected_factor <- detect_outperform_factors(closest_data)$factor[1:30]
  
  # 테스트 데이터 로딩
  test_set <- read.csv(paste0(modeling_path,testing_date[i],"-data.csv"))
  
  # 선택된 팩터로 test set에서 종목을 선정한다
  test_set <- test_set %>% select(code, selected_factor)
  
  selected_stocks <- stocks_for_portfolio(test_set) # 선택된 종목들 
  selected_stocks <- str_pad(selected_stocks , 6, side = c('left'), pad = '0')
  
  # KOR_price 사용해서 해당 시점 선택된 종목들의 값 추출하자
  date_index <- which(ymd(rownames(KOR_price)) %in% testing_date[i]):(which(ymd(rownames(KOR_price)) %in% testing_date[i+1])-1)
  stock_index <- which(colnames(KOR_price) %in% selected_stocks)
  one_month_return <- KOR_price[date_index, stock_index] %>% Return.calculate() %>% slice(-1) %>% sapply(., function(x) {prod(1+x) - 1})
  one_month_return <- as.vector(one_month_return)
  
  ReturnsWithSelectedStocks[[i]] <- one_month_return
}

money <- c(1)

for(i in 1:length(ReturnsWithSelectedStocks)){
  
  tmp <- (money[i]/30) * (ReturnsWithSelectedStocks[[i]]+1) %>% sum()
  money <- c(money, tmp)
  
}

plot(money, type="l")

@ 일단 짜기는 했는데... 결과가 영 엉망이다 ㅠ
@ 개별 팩터로 투자하는 건 오히려 더 별로인가
@ PMI, VIX 등등도 고려해봐야겠다!
