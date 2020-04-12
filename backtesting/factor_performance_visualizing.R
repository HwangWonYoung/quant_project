# import libraries
library("dplyr")
library("ggplot2")
library("gridExtra")
library("grid")

# return performance of specific factor at given dataset
check_factor_return <- function(data, factor){
  
  # select columns
  Y_vars <- c("Y_2_week", "Y_1_month", "Y_2_month", "Y_3_month", "Y_6_month", "Y_12_month")
  data <- data %>% select(factor, Y_vars)
  # make ten portfolio with given factor
  factor_vals <- data[factor] %>% unlist() %>% as.numeric()
  data$portfolio <- cut(factor_vals, quantile(factor_vals, probs=seq(0,1,0.1), include.lowest=TRUE, na.rm=TRUE),
                        labels=1:10, na.rm=TRUE)
  
  data <- data[!is.na(data$portfolio),]
  
  mean_data <- data %>% group_by(portfolio) %>% summarize(count=n(),
                                                          mean_Y_2_w=mean(Y_2_week),
                                                          mean_Y_1_M=mean(Y_1_month),
                                                          mean_Y_2_M=mean(Y_2_month),
                                                          mean_Y_3_M=mean(Y_3_month),
                                                          mean_Y_6_M=mean(Y_6_month),
                                                          mean_Y_12_M=mean(Y_12_month))
  
  sd_data <- data %>% group_by(portfolio) %>% summarize(count=n(),
                                                        sd_Y_2_w=sd(Y_2_week),
                                                        sd_Y_1_M=sd(Y_1_month),
                                                        sd_Y_2_M=sd(Y_2_month),
                                                        sd_Y_3_M=sd(Y_3_month),
                                                        sd_Y_6_M=sd(Y_6_month),
                                                        sd_Y_12_M=sd(Y_12_month))
  
  mean_sd_data <- data %>% group_by(portfolio) %>% summarize(count=n(),
                                                             mean_sd_Y_2_w=mean(Y_2_week)/sd(Y_2_week),
                                                             mean_sd_Y_1_M=mean(Y_1_month)/sd(Y_1_month),
                                                             mean_sd_Y_2_M=mean(Y_2_month)/sd(Y_2_month),
                                                             mean_sd_Y_3_M=mean(Y_3_month)/sd(Y_3_month),
                                                             mean_sd_Y_6_M=mean(Y_6_month)/sd(Y_6_month),
                                                             mean_sd_Y_12_M=mean(Y_12_month)/sd(Y_12_month))
  
  out <- list(mean_data = mean_data, sd_data = sd_data, mean_sd_data = mean_sd_data)
  return(out)
}

# visualize mean return of portfolios
vis_factor_mean_return <- function(data, factor){
  
  data <- check_factor_return(data, factor)$mean_data
  
  g1 <- ggplot(data, aes(x=portfolio, y=mean_Y_2_w)) + geom_bar(stat="identity")
  g2 <- ggplot(data, aes(x=portfolio, y=mean_Y_1_M)) + geom_bar(stat="identity")
  g3 <- ggplot(data, aes(x=portfolio, y=mean_Y_2_M)) + geom_bar(stat="identity")
  g4 <- ggplot(data, aes(x=portfolio, y=mean_Y_3_M)) + geom_bar(stat="identity")
  g5 <- ggplot(data, aes(x=portfolio, y=mean_Y_6_M)) + geom_bar(stat="identity")
  g6 <- ggplot(data, aes(x=portfolio, y=mean_Y_12_M)) + geom_bar(stat="identity")
  
  grid.arrange(g1, g2, g3, g4, g5, g6, ncol=2, top = textGrob(factor ,gp=gpar(fontsize=20,font=3)))
  
}

# visualize standard deviation of return of portfolios
vis_factor_sd_return <- function(data, factor){
  
  data <- check_factor_return(data, factor)$sd_data
  
  g1 <- ggplot(data, aes(x=portfolio, y=sd_Y_2_w)) + geom_bar(stat="identity")
  g2 <- ggplot(data, aes(x=portfolio, y=sd_Y_1_M)) + geom_bar(stat="identity")
  g3 <- ggplot(data, aes(x=portfolio, y=sd_Y_2_M)) + geom_bar(stat="identity")
  g4 <- ggplot(data, aes(x=portfolio, y=sd_Y_3_M)) + geom_bar(stat="identity")
  g5 <- ggplot(data, aes(x=portfolio, y=sd_Y_6_M)) + geom_bar(stat="identity")
  g6 <- ggplot(data, aes(x=portfolio, y=sd_Y_12_M)) + geom_bar(stat="identity")
  
  grid.arrange(g1, g2, g3, g4, g5, g6, ncol=2, top = textGrob(factor ,gp=gpar(fontsize=20,font=3)))
  
}

# visualize mean standard deviation of return of portfolios
vis_factor_mean_sd_return <- function(data, factor){
  
  data <- check_factor_return(data, factor)$mean_sd_data
  
  g1 <- ggplot(data, aes(x=portfolio, y=mean_sd_Y_2_w)) + geom_bar(stat="identity")
  g2 <- ggplot(data, aes(x=portfolio, y=mean_sd_Y_1_M)) + geom_bar(stat="identity")
  g3 <- ggplot(data, aes(x=portfolio, y=mean_sd_Y_2_M)) + geom_bar(stat="identity")
  g4 <- ggplot(data, aes(x=portfolio, y=mean_sd_Y_3_M)) + geom_bar(stat="identity")
  g5 <- ggplot(data, aes(x=portfolio, y=mean_sd_Y_6_M)) + geom_bar(stat="identity")
  g6 <- ggplot(data, aes(x=portfolio, y=mean_sd_Y_12_M)) + geom_bar(stat="identity")
  
  grid.arrange(g1, g2, g3, g4, g5, g6, ncol=2, top = textGrob(factor ,gp=gpar(fontsize=20,font=3)))
  
}