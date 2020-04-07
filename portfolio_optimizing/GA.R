# GA(Genetic Algorithm)을 활용한 개별 팩터 선택(하위 팩터)
# n개의 랜덤한 팩터를 선택, 해당 시점에서 1개월 뒤의 수익률을 최대화하는 팩터의 조합을 유전 알고리즘을 통해 탐색

# pre-defined function
# 아래에 정의된 함수의 최대값을 탐색해나감
custom_fitness <- function(vars, data_x, data_y){
  
  # vars는 1과 0으로 구성된 vector이며 해당 iteration에서 선택된 팩터들을 의미
  names <- colnames(data_x)
  names_2 <- names[vars==1]
  
  # 선택된 팩터로 데이터셋 재구성 
  data_sol <- data_x[, names_2]
  
  # 해당 팩터로 산출된 리턴 값을 계산 
  return_value <- get_return(data_sol, data_y)
  
  return(return_value)
}

# function calculating return
# 주어진 팩터로 상위 30개의 종목을 뽑아 구성한 포트폴리오의 리턴을 반환하는 함수 
get_return <- function(data_tr_sample, target){
  
  stock_return <- target
  factor_score <- data_tr_sample %>% rowSums()
  
  output <- data.frame(stock_return, factor_score) ; colnames(output) <- c("return", "factor_score")
  output <- output %>% arrange(desc(factor_score)) %>% head(30)
  
  final_return <- mean(output$return, na.rm = TRUE)
  
  return(final_return)
  
}

### for loop을 사용해서 해당 시점에 아웃퍼폼한 팩터 조합 탐색 
for(i in 1:length(target_dates)){
  
  dat <- read.csv(paste0(target_dates[i],"-data.csv"), fileEncoding = "CP949", encoding = "UTF-8")
  
  # variables
  Y_vars <- c("Y_2_week", "Y_1_month", "Y_2_month", "Y_3_month", "Y_6_month", "Y_12_month",
              "sharpe_ratio_1m_Y", "sharpe_ratio_3m_Y", "sharpe_ratio_6m_Y")
  
  etc_vars <- c("name", "code", "market", "sector", "price", "market_cap", "size_level")
  
  X_vars <- setdiff(colnames(dat), c(Y_vars, etc_vars))
  
  # 먼저 사용될 Y 변수와 X 변수만으로 구성된 데이터 셋 만들자
  
  dat <- dat %>% select(Y_vars[2], X_vars)
  
  # NA 값 없애자
  # NA 비율 0.2 보다 높은 컬럼 삭제, complete cases만 사용
  
  dat <- dat %>% select(colnames(dat)[sapply(dat, function(x){sum(is.na(x))/length(x)}) <= 0.2])
  dat <- dat[complete.cases(dat),]
  
  Y_col <- "Y_1_month"
  X_cols <- setdiff(colnames(dat), "Y_1_month")
  
  # Z-scoring for each factors
  for(j in X_cols){
    if(j %in% pos_vars){
      dat[j] <- scale(min_rank(desc(dat[j])))
    }else{
      dat[j] <- scale(min_rank(dat[j]))
    }
  }
  
  data_x <- dat %>% select(X_cols)
  data_y <- dat$Y_1_month
  
  # GA parameters
  param_nBits <- ncol(data_x)
  col_names <- colnames(data_x)
  
  # Executing the GA
  ga_GA <- ga(fitness = function(vars) custom_fitness(vars = vars,
                                                      data_x =  data_x,
                                                      data_y = data_y), # custom fitness function
              type = "binary", # optimization data type
              crossover=gabin_uCrossover,  # cross-over method
              elitism = 3, # best N indiv. to pass to next iteration
              pmutation = 0.03, # mutation rate prob
              nBits = param_nBits, # total number of variables
              names=col_names, # variable name
              run=5000, # max iter without improvement (stopping criteria)
              maxiter = 10000, # total runs or generations
              keepBest = TRUE, # keep the best solution at the end
              parallel = T, # allow parallel procesing
              seed = 123 # for reproducibility purposes
  )
  
  result <- ga_GA@solution
  write.csv(result, paste0(target_dates[i],"-ga_result.csv"))
  
}