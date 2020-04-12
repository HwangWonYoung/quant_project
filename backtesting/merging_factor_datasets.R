# load datasets in working directory
temp <- list.files(pattern=("*.csv"))
for (i in 1:length(temp)) assign(str_remove_all(temp[i], ".csv"), read.csv(temp[i], row.names = 1))

# function constructing dataset with given date
return_df_with_given_date <- function(factors, target_date){
  
  # first factor will be standard dataset
  out <- get(factors[1])
  out <- out[which(rownames(out) == target_date),] %>% t() %>% as.data.frame()
  out$code <- rownames(out) ; out <- out[,c(2,1)]
  colnames(out)[2] <- factors[1]
  
  for (i in 2:length(factors)) {
    tmp <- get(factors[i])
    tmp <- tmp[which(rownames(tmp) == target_date),] %>% t() %>% as.data.frame()
    tmp$code <- rownames(tmp) ; tmp <- tmp[,c(2,1)]
    colnames(tmp)[2] <- factors[i]
    out <- out %>% left_join(tmp, by="code")
  }
  
  # rownames should be code
  rownames(out) <- out$code ; out$code <- NULL
  
  return(out)
  
}

# write csv using for loop
for(i in 1:length(target_dates)){
  
  tmp <- return_df_with_given_date(factor_order, target_dates[i])
  write.csv(tmp, paste0(target_dates[i], "-data.csv"))
  
}

# target_dates 는 현재 프로젝트 기준으로 보통 매월 첫 영업일을 사용한다
# return_df_with_given_date 함수를 사용할 때, factors argument에 들어가는 벡터 내의 factor 순서대로 column이 ordering된다 
# return_df_with_given_date 의 factors argument에 각 원소에서 ".csv"를 제거한 temp vector를 인풋으로 사용해도 된다
# i.e. temp <- str_remove(temp, ".csv")
#      return_df_with_given_date(temp, "2020-04-01")