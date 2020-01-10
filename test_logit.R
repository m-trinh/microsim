### Use clean FMLA 2012 data and check coefs of glm 
library(survey)
library(psycho)
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

# x1 <- c(1, 2, 3, 4)
# x2 <- c(10, 20, 30, 40)
# d1 <- data.frame(x1, x2)
# d1 <- d1 %>% psycho::standardize()

# d[Xs] <- d[Xs] %>% psycho::standardize()

for (X in Xs){
  d[paste0('z_', X)] <- d[X] %>% psycho::standardize()
}

# check pre- and post-standardization cols
head(d[, c('empid', 'age', 'z_age')])


## Fit model
d <- head(d, 100)
y <- ys[1]
eq <- as.formula(paste(y, "~", paste(paste0('z_', Xs), collapse="+")))
logit <- glm(eq, family='binomial',data=d)
summary(logit)


####################################

cuse <- read.table("http://data.princeton.edu/wws509/datasets/cuse.dat", 
                   header=TRUE)
attach(cuse)
mod <- glm(cbind(using, notUsing) ~ age + education + wantsMore , family= binomial)
summary(mod)

########
eq <- as.formula(paste(y, "~", paste(Xs, collapse="+")))
mod <- glm(eq , family= binomial, data=d)
summary(mod)
