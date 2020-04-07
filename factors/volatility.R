# calculate daily volatility
# parameters : daily stock price, target date (e.g. first business date of every month),
# year (how long will you consider)
cal_vol_daily <- function(stock_price, target_date, year){
  
  target_date_index <- which(ymd(rownames(stock_price)) %in% ymd(target_date))
  
  out <- data.frame()
  
  for(i in 1:length(target_date_index)){
    
    # slice dataset
    part <- stock_price[(target_date_index[i]-252*year):target_date_index[i],]
    part <- part %>% CalculateReturns()
    part <- apply(part[-1,], 2, sd)
    out <- dplyr::bind_rows(out, part)
  }
  
  colnames(out) <- colnames(stock_price)
  rownames(out) <- rownames(target_date)
  
  return(out)
}

# calculate weekly volatility
# parameters : daily stock price, target date (e.g. first business date of every month),
# year (how long will you consider)
cal_vol_weekly <- function(stock_price, target_date, year){
  
  target_date_index <- which(ymd(rownames(stock_price)) %in% ymd(target_date))
  
  out <- data.frame()
  
  for (i in 1:length(target_date_index)) {
    
    # slice dataset
    part <- stock_price[(target_date_index[i]-252*year):target_date_index[i],]
    part <- part %>% CalculateReturns() %>% apply.weekly(return.cumulative)
    part <- apply(part[-1,], 2, sd)
    out <- dplyr::bind_rows(out, part)
  }
  
  colnames(out) <- colnames(stock_price)
  rownames(out) <- target_date
  
  return(out)
  
}

# why not using PerformanceAnalytics::Return.cumulative?
# it omits every NA values, there is no argument related to NA values
return.cumulative <- function (R, geometric = TRUE){
  if (is.vector(R)) {
    if (!geometric) 
      return(sum(R))
    else {
      return(prod(1 + R) - 1)
    }
  }
  else {
    R = checkData(R, method = "matrix")
    result = apply(R, 2, return.cumulative, geometric = geometric)
    dim(result) = c(1, NCOL(R))
    colnames(result) = colnames(R)
    rownames(result) = "Cumulative Return"
    return(result)
  }
}