ts_closest_dtw <- function(ts, target_date, window_length=30, year=5, top=5 ,plot=FALSE){
  #----------------------------------------------------------------------------------------------------------------------
  # input data : row 는 각 날짜를 의미, column 은 date 와 value 로 이루어져야 하며 column name은 date, close 여야 합니다
  # target_date , window_length : 유사도를 측정하는 데 있어 기준이 되는 패턴을 설정합니다
  # year : target_date 로 부터 몇 년 전 까지의 자료를 고려하는지 설정합니다
  # top : 가장 가까운 n개의 패턴을 추출합니다
  # plot : TRUE 인 경우 plot 까지 반환해줍니다
  #-----------------------------------------------------------------------------------------------------------------------
  if(all(colnames(ts) %in% c("date", "close"))==FALSE){
    stop("need to change column names : date & close")
  }
  
  if(is.numeric(ts$close)==FALSE){
    ts$close <- as.numeric(ts$close) 
  }
  
  # 기준이 되는 window
  standard_index <- which(ymd(ts$date) %in% ymd(target_date))
  standard_window <- ts[(standard_index-window_length):standard_index,]
  
  # 패턴의 유사도를 고려하기 때문에 normalizing을 진행합니다 
  standard_window$close <- (standard_window$close - mean(standard_window$close)) / sd(standard_window$close)
  
  # compare window는 매 월 첫 영업일 기준으로 나뉘어집니다
  # 주어진 time series의 매월 첫 날짜를 추출합니다
  #----------------------------------------------------------    
  Date <- ymd(ts$date)
  Date <- as.data.frame(Date) ; colnames(Date) <- 'date'
  
  first_bizday_of_month <- 
    Date %>% 
    mutate(year = date %>% year, month = date %>% month) %>% 
    group_by(year, month) %>% 
    slice(which.min(date)) %>% data.frame %>% select(date)
  #-----------------------------------------------------------   
  
  if((standard_index - 252*year) < 0){
    compare_window_set <- ts[(1:(standard_index - (window_length + 1))),]
    cat("consider very start of given time series because of shortness")
  } else {
    compare_window_set <- ts[((standard_index - 252*year):(standard_index - (window_length + 1))),]
  }
  
  # 기준 window와 매칭되는 후보 windows
  compare_windows <- list()
  target_indices <- which(ymd(compare_window_set$date) %in% ymd(first_bizday_of_month$date))
  
  target_indices <- target_indices[target_indices > (window_length + 1)]
  
  for(i in 1:length(target_indices)){
    compare_windows[[compare_window_set$date[target_indices[i]]]] <- compare_window_set$close[(target_indices[i]-window_length):(target_indices[i])]
  }
  
  compare_windows <- lapply(compare_windows, function(x)((x-mean(x))/sd(x)))
  
  # dtw를 계산하는 function
  get_dist <- function(query, reference){
    out <- dtw(query, reference, distance.only=TRUE)
    dist <- out$normalizedDistance
    return(dist)
  }
  
  # 기준 window와 비교 window와의 dtw distance를 구합니다 
  result <- lapply(compare_windows, function(x){get_dist(standard_window$close, x)})
  
  # result list를 data.frame으로 변환한 뒤 tidy form으로 변환합니다.
  #---------------------------------------------------------
  result <- unlist(result) %>% as.data.frame()
  result$date <- rownames(result)
  colnames(result)[1] <- "distance"
  result <- result[,c(2,1)]
  rownames(result) <- NULL
  #---------------------------------------------------------
  
  output <- arrange(result, distance)
  output <- output %>% head(top)
  
  # plot argument가 TRUE 일 경우 plot을 반환합니다
  #-------------------------------------------------------------------------------------------------------------------------
  if(plot == TRUE){
    
    ts$group <- "None"
    
    # compare window는 group "compare_window"으로 지정  
    for(i in c(which(ymd(ts$date) %in% ymd(output$date)))){
      ts$group[(i-window_length):i] <- "compare_window"}
    
    # standard window는 group "standard_window"로 지정 
    ts$group[(standard_index-window_length):standard_index] <- "standard_window"
    ts$date <- as.Date(ts$date)
    
    gplot <- ggplot(ts, aes(date, close, group = 1, colour = factor(group))) + 
      geom_line() + scale_color_manual(values = c("None" = "black", "compare_window" = "blue", "standard_window" = "red"))
    
    print(gplot)
    
  }
  #----------------------------------------------------------------------------------------------------------------------------     
  
  return(output)
  
}