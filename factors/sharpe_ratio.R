# calculates a sharpe ratio for one stock
# this function will replace risk free asset(normal case) with market portfolio(KOSPI & KOSDAQ) 
# parameters : market_return(KOSPI & KOSDAQ daily return), stock_return, target_date_index, month
# this case, need to consider KOSPI & KOSDAQ stocks seperately
# need to match date between stock and market
cal_sharpe_ratio <- function(market_return, stock_return, target_date_index, month){
  
  output <- c()
  
  for (i in 1:length(target_date_index)){
    
    # slicing dataset
    stock_return_tmp <- stock_return[(target_date_index[i] - 21*month):target_date_index[i]]
    market_return_tmp <- market_return[(target_date_index[i] - 21*month):target_date_index[i]]
    
    # calculate denominator
    # standard deviation of stock return
    sd_of_stock <- sd(stock_return_tmp)
    
    # calculate numerator
    # return of market index and stock 
    return_diff <- stock_return_tmp - market_return_tmp
    
    sharperatio <- mean(return_diff) / sd_of_stock
    
    output <- c(output, sharperatio)  
  }
  
  return(output)
  
}

# return_sharpe_ratio_set returns total dataset
# parameters : market_return, stock_return(total stock return), target_date, month(how long will you consider)
return_sharpe_ratio_set <- function(market_return, stock_return, target_date, month){
  
  if(all(rownames(market_return)==rownames(stock_return)) == FALSE){
    stop("need to match date between market return and stock return")
  }
  
  target_date_index <- which(ymd(rownames(stock_return)) %in% ymd(target_date))
  
  out <- apply(stock_return, 2, function(x) cal_sharpe_ratio(market_return, x, target_date_index, month))
  out <- as.data.frame(out)
  
  rownames(out) <- head(target_date, length(target_date_index))
  
  return(out)
  
}