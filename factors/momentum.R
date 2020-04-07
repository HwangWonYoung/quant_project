# calculate momentum
# parameters : return(normal return or idiosyncratic return), target_date, month(how long will you consider),
#              ignore_last_month(if it is TRUE, momentum will be calculated without last one month)
cal_momentum <- function(stock_return, target_date, month, ignore_last_month=FALSE){
  
  target_date_index <- which(ymd(rownames(stock_return)) %in% target_date)
  
  output <- data.frame()
  
  for(i in 1:length(target_date_index)){
    # slice dataset
    part <- stock_return[(target_date_index[i]-21*month):target_date_index[i],] 
    
    # if ignore_last_month is TRUE, remove last one month
    if(ignore_last_month == TRUE){
      mnt <- part %>% as.xts() %>% xts::first(21*month - 20) %>% sapply(., function(x) {prod(1+x) - 1})
    }else{
      mnt <- part %>% as.xts()%>% sapply(., function(x) {prod(1+x) - 1})
    }
    output <- dplyr::bind_rows(output, mnt)
  }
  
  rownames(output) <- head(target_date, length(target_date_index))
  return(output)
  
}