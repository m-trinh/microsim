###################################################
# some scrap code for reconciling R results
#
#
#
# chris zhang 8/12/2020
###################################################

library(MASS)
library(dplyr)
## Read in CPS data
fp_cps <- '../data/cps/cps_clean_2014.csv'
cps <- read.csv(fp_cps)

## Run ordinal regression
xvar_formula <- paste('female + black + asian + native + other + age + agesq + BA + GradSch + married + wage12 + wkswork + wkhours + emp_gov',
                      "+ occ_1 + occ_2 + occ_3 + occ_3 + occ_4 +  occ_5 + occ_6 + occ_7 + occ_8",
                      "+ occ_9 + ind_1 + ind_2 + ind_3 + ind_4 + ind_5 + ind_6 + ind_7 + ind_8 + ind_9 + ind_10 + ind_11 + ind_12")

formula <- paste("factor(empsize) ~ ", xvar_formula)
train_filt = "TRUE"
estimate <- polr(as.formula(formula), data = cps %>% filter_(train_filt))
# predict
fp_acs <- '../data/acs/ACS_cleaned_forsimulation_2016_ri_Py.csv'
acs <- read.csv(fp_acs)
Xd <- acs[, c("female", "black",  asian + native + other + age + agesq + BA + GradSch + married + wage12 + wkswork + wkhours + emp_gov)]
polr.predict(estimate, c(1,170))

