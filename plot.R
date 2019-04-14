# This script creates beautiful ensemble graphs of the different parameters 
# the model ran for and saves it as a pdf file

library(ggplot2)

file <- list.files()

dt <- lapply(file, function(x){
  dat <- readLines(x)
  val <- strsplit(dat[1:6]," ")
  vals <- do.call('rbind',lapply(val, function(x){
    as.numeric(tail(x,1))
  }))
  tr_loss <- as.numeric(dat[10:(9+vals[6])])
  val_loss <- as.numeric(dat[(14+vals[6]):(13+vals[6]+vals[6])])
  
  d <- data.frame(obs = 1:vals[6],tr_loss = tr_loss, val_loss = val_loss)
  return(list(d = d, val = vals))
})



plt <- function(dl){
  library(data.table)
  dat <- as.data.table(dl$d)
  dl$val <- dl$val[-c(2,6)]
  nums <- c(dl$val[1], paste(dl$val[2],dl$val[3],sep = ', '), dl$val[4])
  text <- c('LR ','Filters ', 'Seq Size ')
  dat <- dat[1:50,]
  pltdt <-melt(dat[,.(obs,tr_loss,val_loss)],'obs')
  ggplot(data = pltdt, aes(x = obs, y = round(sqrt(value),3), color = variable))+
    geom_line(size = 0.75)+
    theme_minimal()+
    theme(legend.position=c(.80,.70))+
    labs(x = 'Epochs', y = 'RMSE Loss', color ='', subtitle = paste0(text,nums,collapse = ' | '))+
    scale_color_manual(labels = c('Training','Testing'),
                       values=c('red4','navy'))
  
}



l <- lapply(dt, plt)
m1 <- gridExtra::marrangeGrob(l,ncol = 2, nrow = 2)
ggsave("plots.pdf", m1, width=11, height=8.5)