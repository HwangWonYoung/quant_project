cal_return_month <- function(stock_return, target_date, month){
  
  target_date_index <- which(ymd(rownames(stock_return)) %in% target_date)
  
  output <- data.frame()
  
  for(i in 1:(length(target_date_index)-month)){
    part <- stock_return[(target_date_index[i]+1):(target_date_index[i+month]),] # slicing data
    part <- part %>% apply(2, function(x){prod(x+1)-1})
    output <- dplyr::bind_rows(output, part)
  }
  
  rownames(output) <- head(target_date, length(target_date_index)-month)
  return(output)
  
}

cal_return_week <- function(stock_return, target_date, week){
  
  target_date_index <- which(ymd(rownames(stock_return)) %in% target_date)
  
  output <- data.frame()
  
  for(i in 1:(length(target_date_index))){
    part <- stock_return[(target_date_index[i]+1):(target_date_index[i]+(5*week)),] # slicing data
    part <- part %>% apply(2, function(x){prod(x+1)-1})
    output <- dplyr::bind_rows(output, part)
  }
  
  rownames(output) <- head(target_date, length(target_date_index))
  return(output)
  
}
