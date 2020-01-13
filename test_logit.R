### Use clean FMLA 2012 data and check coefs of glm 
library(survey)
library(psycho)
library(data.table)

## set up
setwd("C:\\workfiles\\Microsimulation\\microsim\\")
## Read in data
d <- read.csv("data/fmla_2012/fmla_clean_2012.csv")
# id
id <- 'empid'
# Xs, ys, w
Xs <- c('widowed', 'divorced', 'separated', 'nevermarried',
      'female', 'age','agesq',
      'ltHS', 'someCol', 'BA', 'GradSch',
      'black', 'other', 'asian','native','hisp',
      'nochildren','faminc','coveligd')
ys <- c('take_own', 'take_matdis', 'take_bond', 'take_illchild', 'take_illspouse', 'take_illparent',
        'need_own', 'need_matdis', 'need_bond', 'need_illchild', 'need_illspouse', 'need_illparent',
        'resp_len')
w <- 'weight'
# reduce df
d <- subset(d, select=c(id, Xs, ys, w))

# dropna
d <- na.omit(d)

## Standardize Xs

# NOTE: psycho::standardize() does not affect binary vars, so do it manually. R sd uses n-1 as divisor.
# d[Xs] <- d[Xs] %>% psycho::standardize()

for (X in Xs){
  #d[paste0('z_', X)] <- d[X] %>% psycho::standardize() # this will not affect binary cols
  d[paste0('z_', X)] <- (d[X] - mean(d[, X]))/sd(d[, X])

}
z_Xs <- paste0('z_', Xs)

# check pre- and post-standardization cols
# head(d[, c('empid', 'age', 'z_age', 'female', 'z_female')])


## Fit model

fit_logit <- function(d, Xs, w, y, standardized, weighted){
  
  # use standardized xvars if opted
  if (standardized){
    xvars <- z_Xs
  }
  else{
    xvars <- Xs
  }
  
  # if matdis, reduce to female only rows, remove female from xvar
  if (is.element(y, c('take_matdis', 'need_matdis'))){
    d <- subset(d, female==1)
    if (standardized){
      xvars <- z_Xs[z_Xs!='z_female']
    }
    else{
      xvars <- Xs[Xs!='female']
    }
  }
  
  # fit model
  eq <- as.formula(paste(y, "~", paste(xvars, collapse="+")))
  
  if (weighted==FALSE){
    logit <- glm(eq, family='quasibinomial',data=d)
  }
  else{
    des <- svydesign(id = ~empid,  weights = ~weight, data = d)
    logit <- svyglm(as.formula(eq), data = d,
                     family = "quasibinomial",design = des)
  }
  
  # get bs, phats
  bs <- summary(logit)$coefficients[, 1] # coefs
  phats <- head(predict(logit, d[, xvars], type='response'), 10) # phats
  
  # list to return
  li <- list('bs'=bs, 'phats'=phats)
  
  return (li)
}

# # check function works
# y <- 'take_own'
# #y <-'take_matdis'
# li <-fit_logit(d, Xs, w, y, standardized=TRUE, weighted=FALSE)
# print(li$bs)
# print(li$phats)

# make dfs for bs phats
Dbs <- data.frame()
Dps <- data.frame()
for (y in ys){
  li<- fit_logit(d, Xs, w, y, standardized=TRUE, weighted=TRUE)
  
  dbs <- data.frame(li$bs)
  names(dbs)[1] <- y
  if (dim(Dbs)[1]==0){ # empty Dbs
    Dbs <- transform(merge(Dbs, dbs, by=0, all=TRUE), row.names=Row.names, Row.names=NULL)
  }
  else{ # non-empty Dbs, use all.x=TRUE
    Dbs <- transform(merge(Dbs, dbs, by=0, all.x=TRUE), row.names=Row.names, Row.names=NULL)
  }
  
  dps <- data.frame(li$phats)
  names(dps)[1] <- y
  if (dim(Dps)[1]==0){ # empty Dps
    Dps <- transform(merge(Dps, dps, by=0, all=TRUE), row.names=Row.names, Row.names=NULL)
  }
  else{ # non-empty Dps, use all.x=TRUE
    Dps <- transform(merge(Dps, dps, by=0, all.x=TRUE), row.names=Row.names, Row.names=NULL)

  }  
  # order by numeric index
  Dps$index <- as.numeric(row.names(Dps))
  Dps <- Dps[order(Dps$index), ]
  Dps$index <- NULL
}

# output
write.csv(Dbs, "./PR_comparison/R_bs.csv")
write.csv(Dps, "./PR_comparison/R_phats.csv")

















