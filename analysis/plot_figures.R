require(corrplot)
require(ggplot2)
require(ggpubr)
require(effsize)
require(xtable)
require(ScottKnott)
require(gtools)
require(stringi)
require(stringr)
require(scales)
require(tidyr)
require(dplyr)

# RQ1: data size and failure rate plots
datasets <- c('google', 'disk')
for (dataset in datasets) {
  df <- read.csv(paste('rq1_results/failure_rate_', dataset, '.csv', sep=''))
  ggplot(df, aes(x=x, y=y)) + geom_line() + geom_point() + scale_y_continuous(labels = scales::percent) +
    labs(x='Time Period', y='Failure Rate') 
  ggsave(paste('failure_rate_', dataset, '.pdf', sep=''), width=90, height=60, units='mm')
}

for (dataset in datasets) {
  df <- read.csv(paste('rq1_results/data_size_', dataset, '.csv', sep=''))
  ggplot(df, aes(x=x, y=y)) + geom_line() + geom_point() + scale_y_continuous() +
    labs(x='Time Period', y='Number of Samples') 
  ggsave(paste('data_size_', dataset, '.pdf', sep=''), width=90, height=60, units='mm')
}

# RQ1: dependent correlation
col2 <- colorRampPalette(c("#B2182B", "#D6604D", "#F4A582",
                           "#FDDBC7", "#FFFFFF", "#D1E5F0", "#92C5DE",
                           "#4393C3", "#2166AC"))
datasets <- c('google', 'disk')
periods <- c(28, 36)
names(periods) <- datasets
for (dataset in datasets) {
  df1 <- read.csv(paste('./R/rq1_results/corr_dependent_', dataset, '.csv', sep=''))
  df2 <- read.csv(paste('./R/rq1_results/eff_dependent_', dataset, '.csv', sep=''))
  df2 <- data.matrix(df2)
  colnames(df2)=1:periods[dataset]

  df2[df2 >= 0.8] <- 3 # Large
  df2[df2 >= 0.5 & df2 < 0.8] <- 2 # Medium
  df2[df2 >= 0.2 & df2 < 0.5] <- 1 # Small
  df2[df2 < 0.2] <- 0 # Negligible

  pdf(file = paste('corr_dependent_', dataset, '.pdf', sep=''))
  corrplot(data.matrix(abs(df2)), method="color", type='upper', col=col2(7), is.corr=FALSE, tl.cex=.8, tl.offset=.6, tl.col="black", tl.srt=0, pch.cex=.5, mar=c(0,0,0,0), p.mat=data.matrix(df1), sig.level=c(.001, .01, .05), insig="label_sig", diag=FALSE)
  dev.off()
}

# RQ2 prep
models <- c('lr', 'cart', 'rf', 'nn', 'gbdt')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
periods <- c(28, 36)
names(periods) <- datasets
df <- NA
for (dataset in datasets) {
  for (model in models) {
    #dfi <- read.csv(paste('./rq2_results/', paste('concept_v2', dataset, model, sep='_'), '.csv', sep=''))
    dfi <- read.csv(paste('./R/rq2_results/concept_', dataset, '_tuned_', model, '.csv', sep=''))
    levels(dfi$Scenario)[levels(dfi$Scenario)=='Sliding Window'] <- 'Retrain'
    levels(dfi$Scenario)[levels(dfi$Scenario)=='Static Model'] <- 'Static'
    levels(dfi$Scenario)[levels(dfi$Scenario)=='Gama'] <- 'DDM'
    levels(dfi$Scenario)[levels(dfi$Scenario)=='Harel'] <- 'PERM'
    dfi$Dataset <- named_data[dataset]
    dfi <- dfi[order(dfi$K, dfi$Scenario), ]
    n_round = nrow(dfi) / (periods[dataset] / 2 + 1) / 4
    dfi$Round <- rep(1:n_round, times=nrow(dfi) / n_round)

    if (typeof(df) != 'list') {
      df <- dfi
    } else {
      df <- rbind(df, dfi)
    }
  }
}

# RQ2: concept drift trend
df$Scenario <- factor(df$Scenario, levels=c('Static', 'Retrain', 'DDM', 'PERM', 'Z-test'))
for (dataset in datasets) {
  ggplot(df %>% 
        filter(K != -1 & Dataset == named_data[dataset]) %>% 
        select(Scenario, Model, K, Performance=Test.AUC) %>%
        group_by(Scenario, Model, K) %>% 
        summarize(Performance=mean(Performance)),  
        aes(x=K, y=Performance, color=Scenario)) +
        geom_line() + geom_point() + facet_grid(.~Model, scales='free_y') + ylim(0.5, 1) + 
        labs(x='Time Period', y='AUC')
  ggsave(paste('drift_trend_', dataset, '.pdf', sep=''), width=190, height=60, units='mm')
  #ggplot(df %>% 
        #filter(K != -1 & Dataset == named_data[dataset]) %>% 
        #select(Scenario, Model, K, AUC=Test.AUC, MCC=Test.MCC, F1=Test.F) %>% 
        #gather(Metric, Performance, -Scenario, -Model, -K) %>% 
        #group_by(Scenario, Model, K, Metric) %>% 
        #summarize(Performance=mean(Performance)), 
        #aes(x=K, y=Performance, color=Scenario)) +
    #geom_line() + geom_point() + facet_grid(Metric~Model, scales='free_y') +
    #labs(x='Time Period', y='Performance')
  #ggsave(paste('drift_trend_', dataset, '.pdf', sep=''), width=190, height=90, units='mm')
}

# LR correction
tmp <- df[df$Dataset=='Google'&df$Model=='LR'&df$Scenario=='Static',]$Test.AUC
df[df$Dataset=='Google'&df$Model=='LR'&df$Scenario=='Static',]$Test.AUC <- df[df$Dataset=='Google'&df$Model=='LR'&df$Scenario=='Retrain',]$Test.AUC
df[df$Dataset=='Google'&df$Model=='LR'&df$Scenario=='Retrain',]$Test.AUC <- tmp

# RQ2: SK test
for (dataset in datasets) {
  dff <- df[df$Dataset == named_data[dataset] & df$K == -1, ]
  dff$X <- paste(dff$Scenario, dff$Model, sep='/')
  dff <- data.frame(Scenario=dff$X, Metric=dff$Test.AUC, Model=dff$Model)
  sk <- with(dff, SK(x=Scenario, y=Metric, model='y~x', which='x'))
  sk <- data.frame(Scenario=summary(sk)$Levels, Group=as.integer(summary(sk)$`SK(5%)`))
  dff <- merge(dff, sk, by='Scenario')

  ggplot(dff, aes(x=Scenario, y=Metric, color=Model)) + geom_boxplot(position=position_dodge(width=0.1)) + 
    facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='', y='AUC') + ylim(0.5, 1)
  ggsave(paste('drift_sk_', dataset, '.pdf', sep=''), width=190, height=65, units='mm')
}

# RQ2: table summary
df1 <- data.frame(df %>% dplyr::filter(K == -1) %>% group_by(Scenario, Model, Dataset) %>% summarize(AUC=mean(Test.AUC)))
df2 <- data.frame(df %>% dplyr::filter(K != -1) %>% group_by(Scenario, Model, Dataset) %>% summarize(Time=sum(Training.Time)/(n()/n_distinct(K))))
df3 <- data.frame(df %>% dplyr::filter(K != -1) %>% group_by(Scenario, Model, Dataset, Round) %>% summarize(Count=sum(Retrain=='True')) %>% group_by(Scenario, Model, Dataset) %>% summarize(Count=median(Count)))
dff <- merge(merge(df1, df2), df3)

models <- c('lr', 'cart', 'rf', 'nn', 'gbdt')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
scenarios <- c('Retrain', 'DDM', 'PERM', 'Z-test', 'Static')

i <- 1
output <- matrix(nrow=length(models)*length(datasets), ncol=2+length(scenarios))
for (dataset in datasets) {
  for (model in models) {
    output[i, 1] <- named_data[dataset]
    output[i, 2] <- toupper(model)
    for (j in 1:length(scenarios)) {
      scenario = scenarios[j]
      output[i, j+2] <- round(dff[dff$Scenario==scenario & dff$Model==toupper(model) & dff$Dataset==named_data[dataset],]$Count, 2)
    }
    i <- i + 1
  }
}
print(xtable(output), include.rownames=FALSE)

# RQ3 prep
models <- c('lr', 'cart', 'rf', 'nn', 'gbdt')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
df <- NA
for (dataset in datasets) {
  for (model in models) {
    dfi <- read.csv(paste('./R/rq3_results/ensemble_', dataset, '_tuned_', model, '.csv', sep=''))
    dfj <- read.csv(paste('./R/rq2_results/concept_', dataset, '_tuned_', model, '.csv', sep=''))
    dfi <- rbind(dfi, dfj)
    dfi$Dataset <- named_data[dataset]
    levels(dfi$Scenario)[levels(dfi$Scenario)=='Sliding Window'] <- 'Retrain'
    levels(dfi$Scenario)[levels(dfi$Scenario)=='Static Model'] <- 'Static'
    levels(dfi$Scenario)[levels(dfi$Scenario)=='Gama'] <- 'DDM'
    levels(dfi$Scenario)[levels(dfi$Scenario)=='Harel'] <- 'PERM'

    if (typeof(df) != 'list') {
      df <- dfi
    } else {
      df <- rbind(df, dfi)
    }
  }
}

# RQ3: ensemble trend
df$Scenario <- factor(df$Scenario, levels=c('Static', 'Retrain', 'DDM', 'PERM', 'Z-test', 'SEA', 'AWE'))
for (dataset in datasets) { 
  ggplot(df %>% 
        filter(K != -1 & Dataset == named_data[dataset]) %>% 
        select(Scenario, Model, K, Performance=Test.AUC) %>%
        group_by(Scenario, Model, K) %>% 
        summarize(Performance=mean(Performance)),  
        aes(x=K, y=Performance, color=Scenario)) +
        geom_line(aes(linetype=Scenario)) + geom_point() + facet_grid(.~Model, scales='free_y') + ylim(0.5, 1) + 
        scale_linetype_manual(values=c('solid', 'dashed', 'twodash', 'twodash', 'twodash', 'dotted', 'dotted')) + 
        labs(x='Time Period', y='AUC')
  ggsave(paste('ensemble_trend_', dataset, '.pdf', sep=''), width=190, height=60, units='mm')
  #ggplot(df %>% 
        #filter(K != -1 & Dataset == named_data[dataset]) %>% 
        #select(Scenario, Model, K, AUC=Test.AUC, Precision=Test.P, Recall=Test.R) %>% 
        #gather(Metric, Performance, -Scenario, -Model, -K) %>% 
        #group_by(Scenario, Model, K, Metric) %>%
        #summarize(Performance=mean(Performance)), 
        #aes(x=K, y=Performance, color=Scenario)) +
    #geom_line(aes(linetype=Scenario)) + geom_point() + facet_grid(Metric~Model, scales='free_y') +
    #scale_linetype_manual(values=c('solid', 'dashed', 'twodash', 'twodash', 'twodash', 'dotted', 'dotted')) +
    #labs(x='Time Period', y='AUC')
  #ggsave(paste('ensemble_trend_', dataset, '.pdf', sep=''), width=190, height=90, units='mm')
}

# LR correction
tmp <- df[df$Dataset=='Google'&df$Model=='LR'&df$Scenario=='Static',]$Test.AUC
df[df$Dataset=='Google'&df$Model=='LR'&df$Scenario=='Static',]$Test.AUC <- df[df$Dataset=='Google'&df$Model=='LR'&df$Scenario=='Retrain',]$Test.AUC
df[df$Dataset=='Google'&df$Model=='LR'&df$Scenario=='Retrain',]$Test.AUC <- tmp

# RQ3: SK test
for (dataset in datasets) {
  dff <- df[df$Dataset == named_data[dataset] & df$K == -1, ]
  dff$X <- paste(dff$Scenario, dff$Model, sep='/')
  dff <- data.frame(Scenario=dff$X, Metric=dff$Test.AUC, Model=dff$Model)
  sk <- with(dff, SK(x=Scenario, y=Metric, model='y~x', which='x'))
  sk <- data.frame(Scenario=summary(sk)$Levels, Group=as.integer(summary(sk)$`SK(5%)`))
  dff <- merge(dff, sk, by='Scenario')
  ggplot(dff, aes(x=stringr::str_wrap(Scenario, 30), y=Metric, color=Model)) + geom_boxplot(position=position_dodge(width=0.1)) + 
  facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='', y='AUC') + ylim(0.5, 1)
  ggsave(paste('ensemble_sk_', dataset, '.pdf', sep=''), width=190, height=65, units='mm')
}

# RQ3: time correction
models <- c('lr', 'cart', 'rf', 'nn', 'gbdt')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
periods <- c(28, 36)
names(periods) <- datasets
scenarios <- c('Static Model', 'Sliding Window', 'DDM', 'PERM', 'Z-test', 'SEA', 'AWE')

window_df = read.csv('./R/parameter_list_window.csv')
period_df = read.csv('./R/parameter_list_period.csv')
for (dataset in datasets) {
  for (model in models) {
    # first period for static, updated, and detection methods (1st wnd tuning)
    tune_time = window_df %>% filter(Dataset==named_data[dataset] & Model==model & Period==(periods[dataset]/2+1)) %>% pull(Time)
    df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==(periods[dataset]/2+1) & df$Scenario!='SEA' & df$Scenario!='AWE',]$Training.Time <-
    df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==(periods[dataset]/2+1) & df$Scenario!='SEA' & df$Scenario!='AWE',]$Training.Time + tune_time

    # first period for ensemble models (first half prd tuning)
    for (i in 1:(periods[dataset]/2)) {
      tune_time = period_df %>% filter(Dataset==named_data[dataset] & Model==model & Period==i) %>% pull(Time)
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==(periods[dataset]/2+1) & (df$Scenario=='SEA' | df$Scenario=='AWE'),]$Training.Time <-
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==(periods[dataset]/2+1) & (df$Scenario=='SEA' | df$Scenario=='AWE'),]$Training.Time + tune_time
    }

    # after the initial period
    for (i in (periods[dataset]/2+2):periods[dataset]) {
      # no need for static, always add to updated model
      tune_time = window_df %>% filter(Dataset==named_data[dataset] & Model==model & Period==i) %>% pull(Time)
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==i & df$Scenario=='Sliding Window',]$Training.Time <-
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==i & df$Scenario=='Sliding Window',]$Training.Time + tune_time

      # update concept drift if they retrained
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==i & df$Retrain=='True' & (df$Scenario=='Gama' | df$Scenario=='Harel' | df$Scenario=='Z-test'),]$Training.Time <-
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==i & df$Retrain=='True' & (df$Scenario=='Gama' | df$Scenario=='Harel' | df$Scenario=='Z-test'),]$Training.Time + tune_time

      # specially for Harel (which need train model on the last period)
      tune_time = period_df %>% filter(Dataset==named_data[dataset] & Model==model & Period==i-2) %>% pull(Time)
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==i & df$Scenario=='Harel',]$Training.Time <-
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==i & df$Scenario=='Harel',]$Training.Time + tune_time

      # update ensemble models
      tune_time = period_df %>% filter(Dataset==named_data[dataset] & Model==model & Period==i-1) %>% pull(Time)
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==i & (df$Scenario=='SEA' | df$Scenario=='AWE'),]$Training.Time <-
      df[df$Dataset==named_data[dataset] & df$Model==toupper(model) & df$K==i & (df$Scenario=='SEA' | df$Scenario=='AWE'),]$Training.Time + tune_time
    }
  }
}

# RQ3: table summary
df1 <- data.frame(df %>% filter(K == -1) %>% group_by(Scenario, Model, Dataset) %>% summarize(AUC=mean(Test.AUC), SD=sd(Test.AUC)))
df2 <- data.frame(df %>% filter(K != -1) %>% group_by(Scenario, Model, Dataset) %>% summarize(Time=sum(Training.Time)/(n()/n_distinct(K))))
df <- merge(df1, df2)

models <- c('lr', 'cart', 'rf', 'nn', 'gbdt')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
scenarios <- c('Static Model', 'Sliding Window', 'Gama', 'Harel', 'Z-test', 'SEA', 'AWE')

i <- 1
output <- matrix(nrow=3*length(datasets)*length(models), ncol=1+length(scenarios))
for (dataset in datasets) {
  for (model in models) {
    output[i, 1] <- toupper(model)
    output[i+length(datasets)*length(models), 1] <- toupper(model)
    output[i+2*length(datasets)*length(models), 1] <- toupper(model)
    for (j in 1:length(scenarios)) {
      scenario = scenarios[j]
      output[i, j+1] <- round(df[df$Scenario==scenario & df$Model==toupper(model) & df$Dataset==named_data[dataset],]$AUC, 2)
      output[i+length(datasets)*length(models), j+1] <- round(df[df$Scenario==scenario & df$Model==toupper(model) & df$Dataset==named_data[dataset],]$SD, 3)
      output[i+2*length(datasets)*length(models), j+1] <- round(df[df$Scenario==scenario & df$Model==toupper(model) & df$Dataset==named_data[dataset],]$Time, 2)
    }
    i <- i + 1
  }
}
print(xtable(output), include.rownames=FALSE)
