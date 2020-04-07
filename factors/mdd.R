# calculate MDD
# parameters : stock_price, target_date_index, month(how long will you consider)
# cal_mdd function applies to each stock
cal_mdd <- function(stock_price, target_date_index, month){
  
  out <- c()
  
  for(i in 1:length(target_date_index)){
    # slice dataset
    part <- stock_price[(target_date_index[i]-(21*month)):target_date_index[i]]
    max_val <- max(part)
    min_val <- min(part)
    mdd <- (min_val - max_val) / max_val
    out <- c(out, mdd)
  }
  
  return(out)
  
}

# return_mdd_set returns total dataset
# parameters : stock_price(total stock price), target_date, month(how long will you consider)
return_mdd_set <- function(stock_price, target_date, month){
  
  target_date_index <- which(ymd(rownames(stock_price)) %in% ymd(target_date))
  
  out <- apply(stock_price, 2, function(x) cal_mdd(x, target_date_index, month))
  out <- as.data.frame(out)
  
  rownames(out) <- head(target_date, length(target_date_index))
  
  return(out)
  
}