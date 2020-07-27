if( ! require( 'R.utils' , character.only = TRUE ) ){
      #  If package was not able to be loaded then re-install
      install.packages( 'R.utils' , dependencies = TRUE )
      #  Load package after installing
      require( 'R.utils' , character.only = TRUE )
    }

source("0_master_execution_function.R")
args = commandArgs(asValue=TRUE)
keys <- attachLocally(args)

d <- policy_simulation(
	acs_dir=acs_dir,
	fmla_file=fmla_file,
	fmla_year=fmla_year,
	acs_year=acs_year,
	saveCSV=TRUE,
	FEDGOV=as.logical(FEDGOV),
	STATEGOV=as.logical(STATEGOV),
	LOCALGOV=as.logical(LOCALGOV),
	xvars=c("widowed", "divorced", "separated", "nevermarried", "female",
		  'age_cat', "ltHS", "someCol", "BA", "GradSch", "black",
		  "other", "asian",'native', "hisp","nochildren",'faminc_cat','coveligd'),
	base_bene_level=as.numeric(base_bene_level),
	impute_method=gsub('_', ' ', impute_method),
	makelog = TRUE,
	state=state,
	SELFEMP=as.logical(SELFEMP),
	place_of_work=as.logical(place_of_work),
	clone_factor=as.numeric(clone_factor),
	week_bene_cap=as.numeric(week_bene_cap),
	own_uptake=as.numeric(own_uptake),
	matdis_uptake=as.numeric(matdis_uptake),
	bond_uptake=as.numeric(bond_uptake),
	illparent_uptake=as.numeric(illparent_uptake),
	illspouse_uptake=as.numeric(illspouse_uptake),
	illchild_uptake=as.numeric(illchild_uptake),
	maxlen_own=as.numeric(maxlen_own),
	maxlen_matdis=as.numeric(maxlen_matdis),
	maxlen_bond=as.numeric(maxlen_bond),
	maxlen_illparent=as.numeric(maxlen_illparent),
	maxlen_illspouse=as.numeric(maxlen_illspouse),
	maxlen_illchild=as.numeric(maxlen_illchild),
	maxlen_total=as.numeric(maxlen_total),
	maxlen_PFL=as.numeric(maxlen_PFL),
	maxlen_DI=as.numeric(maxlen_DI),
	earnings=as.numeric(earnings),
	weeks=as.numeric(weeks),
	ann_hours=as.numeric(ann_hours),
	minsize=as.numeric(minsize),
	random_seed=as.numeric(random_seed),
	progress_file=progress_file,
	log_directory=log_directory,
	out_dir=out_dir,
	output_stats=c('standard', 'state_compar'),
	kval=5,
	alpha=as.numeric(alpha),
	wait_period=wait_period,
	wait_period_recollect=wait_period_recollect,
	dual_receiver=dual_receiver,
	min_takeup_cpl=min_takeup_cpl,
	ABF_enabled=FALSE
)
