# calculating idiosyncratic return function
# parameters : market(KOSPI & KOSDAQ daily return), stock(stock return), year (how long will you consider)
# cal_idiosyncratic_return function applies to each stock
cal_idiosyncratic_return <- function(market, stock, year){
  
  output <- c()
  
  for(i in 1:(length(market)-(252*year))){
    
    input_X = market[i:(i+252*year)]  
    input_Y = stock[i:(i+252*year)]
    
    out <- tryCatch(
      lm(input_Y~input_X, na.action = na.fail),
      error = function(e){return(NA)}
    )
    
    if(is.object(out)==TRUE){
      output <- c(output, tail(out$residuals, 1))
    } else {
      output <- c(output, out)
    }
    
  }
  return(output)
}

# return_idiosyncratic_set returns total dataset
# parameters : stock_return (total stock return, column indicates each stock),
#              market(KOSPI & KOSDAQ daily return), year (how long will you consider)
return_idiosyncratic_set <- function(stock_return, market_return, year){
  
  out <- apply(stock_return, 2, function(x) cal_idiosyncratic_return(market_return, x, year))
  out <- as.data.frame(out)
  
  colnames(out) <- colnames(stock_return)
  rownames(out) <- rownames(stock_return) %>% tail(nrow(out))
  
  return(out)
  
}