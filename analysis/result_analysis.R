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

# data size and failure rate plots
datasets <- c('google', 'disk')
for (dataset in datasets) {
  df <- read.csv(paste('preliminary_results/failure_rate_', dataset, '.csv', sep=''))
  ggplot(df, aes(x=x, y=y)) + geom_line() + geom_point() + scale_y_continuous(labels = scales::percent) +
    labs(x='Time Period', y='Failure Rate') 
  ggsave(paste('failure_rate_', dataset, '.pdf', sep=''), width=90, height=60, units='mm')
}

for (dataset in datasets) {
  df <- read.csv(paste('preliminary_results/data_size_', dataset, '.csv', sep=''))
  ggplot(df, aes(x=x, y=y)) + geom_line() + geom_point() + scale_y_continuous() +
    labs(x='Time Period', y='Number of Samples') 
  ggsave(paste('data_size_', dataset, '.pdf', sep=''), width=90, height=60, units='mm')
}

# dependent correlation
col2 <- colorRampPalette(c("#B2182B", "#D6604D", "#F4A582",
                           "#FDDBC7", "#FFFFFF", "#D1E5F0", "#92C5DE",
                           "#4393C3", "#2166AC"))
datasets <- c('google', 'disk')
periods <- c(28, 36)
names(periods) <- datasets
for (dataset in datasets) {
  df1 <- read.csv(paste('preliminary_results/corr_dependent_', dataset, '.csv', sep=''))
  df2 <- read.csv(paste('preliminary_results/eff_dependent_', dataset, '.csv', sep=''))
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
df1 <- read.csv('experiment_results/drift_detection_model_results.csv')
df2 <- read.csv('experiment_results/ensemble_model_results.csv')
df3 <- read.csv('experiment_results/online_model_results.csv')
df <- rbind(df1, df2, df3)
df$Dataset <- factor(df$Dataset, levels=c('Google', 'Backblaze'))
df$Model <- factor(df$Model, levels=c('LR', 'CART', 'RF', 'NN', 'GBDT', 'Online'))
df$Scenario <- factor(df$Scenario, levels=c('Stationary', 'Retrain', 'DDM', 'PERM', 'STEPD', 'SEA', 'AWE', 'HT', 'ARF', 'AUE'))

datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
periods <- c(28, 36)
names(periods) <- datasets

# trend figure for AUC performance
for (dataset in datasets) { 
  ggplot(df %>% 
        filter(Testing.Period > periods[dataset]/2 & Dataset == named_data[dataset]) %>% 
        select(Scenario, Model, Testing.Period, Performance=Test.AUC) %>%
        group_by(Scenario, Model, Testing.Period) %>% 
        summarize(Performance=mean(Performance)),  
        aes(x=Testing.Period, y=Performance, color=Scenario)) +
        geom_line(aes(linetype=Scenario)) + geom_point() + facet_grid(.~Model, scales='free_y') + ylim(0.5, 1) + 
        scale_linetype_manual(values=c('solid', 'solid', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted', 'twodash', 'twodash', 'twodash')) + 
        labs(x='Testing Time Period', y='AUC')
  ggsave(paste('auc_trend_', dataset, '.pdf', sep=''), width=190, height=75, units='mm')
}

# SK test for AUC performance
for (dataset in datasets) {
  df_sk <- df %>% filter(Dataset==named_data[dataset] & Testing.Period == -1)
  df_sk <- data.frame(Key=paste(df_sk$Scenario, df_sk$Model, sep='/'), 
                      Value=df_sk$Test.AUC,
                      Model=df_sk$Model)
  sk <- with(df_sk, SK(x=Key, y=Value, model='y~x', which='x'))
  sk <- summary(sk)
  df_sk <- merge(df_sk, data.frame(Key=sk$Levels, Group=as.integer(sk$`SK(5%)`)), by='Key')
  ggplot(df_sk, aes(x=stringr::str_wrap(Key, 30), y=Value, color=Model)) + geom_boxplot(position=position_dodge(width=0.1)) + 
  facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='', y='AUC') + ylim(0.5, 1)
  ggsave(paste('auc_sk_', dataset, '.pdf', sep=''), width=190, height=65, units='mm')
}

# aggregate data to each round
# Dataset, Scenario, Model, Round, Training Time, Testing Time, AUC, Retrain Counts
df1 <- df %>% group_by(Dataset, Scenario, Model, Round) %>% filter(Testing.Period != -1) %>% summarize(Training.Time=sum(Training.Time), Testing.Time=sum(Testing.Time), SD=sd(Test.AUC))
df2 <- df %>% group_by(Dataset, Scenario, Model, Round) %>% filter(Testing.Period != -1, Retrain=='True') %>% summarize(Retrain.Count=n())
df3 <- df %>% filter(Testing.Period == -1) %>% select(Dataset, Scenario, Model, Round, AUC=Test.AUC)
df_round <- merge(df1, df2, all=T, by=c('Dataset', 'Scenario', 'Model', 'Round'))
df_round <- merge(df_round, df3, all=T, by=c('Dataset', 'Scenario', 'Model', 'Round'))
df_round[is.na(df_round)] <- 0

# SK test for training+testing each *round*
for (dataset in datasets) {
  df_sk <- df_round %>% filter(Dataset==named_data[dataset])
  df_sk <- data.frame(Key=paste(df_sk$Scenario, df_sk$Model, sep='/'), 
                      Value=df_sk$Training.Time+df_sk$Testing.Time,
                      Model=df_sk$Model)
  sk <- with(df_sk, SK(x=Key, y=Value, model='y~x', which='x'))
  sk <- summary(sk)
  df_sk <- merge(df_sk, data.frame(Key=sk$Levels, Group=as.integer(sk$`SK(5%)`)), by='Key')
  
  ggplot(df_sk, aes(x=stringr::str_wrap(Key, 30), y=Value, color=Model)) + 
    geom_boxplot(position=position_dodge(width=0.1)) + 
    facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='', y='Training+Testing Time')
  ggsave(paste('training_testing_time_sk_', dataset, '.pdf', sep=''), width=190, height=65, units='mm')
}

# SK test for SD each *round*
for (dataset in datasets) {
  df_sk <- df_round %>% filter(Dataset==named_data[dataset])
  df_sk <- data.frame(Key=paste(df_sk$Scenario, df_sk$Model, sep='/'), 
                      Value=df_sk$SD,
                      Model=df_sk$Model)
  sk <- with(df_sk, SK(x=Key, y=Value, model='y~x', which='x'))
  sk <- summary(sk)
  df_sk <- merge(df_sk, data.frame(Key=sk$Levels, Group=as.integer(sk$`SK(5%)`)), by='Key')
  
  ggplot(df_sk, aes(x=stringr::str_wrap(Key, 30), y=Value, color=Model)) + 
    geom_boxplot(position=position_dodge(width=0.1)) + 
    facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='', y='Standard Deviation')
  ggsave(paste('sd_sk_', dataset, '.pdf', sep=''), width=190, height=65, units='mm')
}

# we cannot calculate EC ratio on the online models
df_round_wo_online <- df_round[df_round$Model != 'Online', ]
# Dataset, Scenario, Model, Round, AUC.Imp
df_ec <- vector(mode='double', length=nrow(df_round_wo_online))
for (i in 1:nrow(df_round_wo_online)) {
  dataset <- df_round_wo_online[i, 1]
  model <- df_round_wo_online[i, 3]
  round <- df_round_wo_online[i, 4]
  auc <- df_round_wo_online[i, 9]
  stationary_auc <- df_round %>% filter(Dataset==dataset & Model==model & Round==round & Scenario=='Stationary') %>% pull(AUC)
  df_ec[i] <- (auc - stationary_auc) / stationary_auc
}
df_round_wo_online$AUC.Imp <- df_ec
df_round_wo_online$EC <- df_ec*100/df_round_wo_online$Retrain

# SK test for EC ratio each *round*
for (dataset in datasets) {
  df_sk <- df_round_wo_online %>% filter(Dataset==named_data[dataset])
  df_sk <- data.frame(Key=paste(df_sk$Scenario, df_sk$Model, sep='/'), 
                      Value=df_sk$EC,
                      Model=df_sk$Model)
  df_sk <- na.omit(df_sk)
  sk <- with(df_sk, SK(x=Key, y=Value, model='y~x', which='x'))
  sk <- summary(sk)
  df_sk <- merge(df_sk, data.frame(Key=sk$Levels, Group=as.integer(sk$`SK(5%)`)), by='Key')
  
  ggplot(df_sk, aes(x=stringr::str_wrap(Key, 30), y=Value, color=Model)) + 
    geom_boxplot(position=position_dodge(width=0.1)) + 
    facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='', y='EC Ratio')
  ggsave(paste('ec_ratio_sk_', dataset, '.pdf', sep=''), width=190, height=65, units='mm')
}

# Table summary
models <- c('lr', 'cart', 'rf', 'nn', 'gbdt')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
scenarios <- c('Stationary', 'Retrain', 'DDM', 'PERM', 'STEPD', 'SEA', 'AWE')

i <- 1
output <- matrix(nrow=4*length(datasets)*length(models), ncol=1+length(scenarios))
for (dataset in datasets) {
  for (model in models) {
    output[i, 1] <- toupper(model)
    output[i+length(datasets)*length(models), 1] <- toupper(model)
    output[i+2*length(datasets)*length(models), 1] <- toupper(model)
    output[i+3*length(datasets)*length(models), 1] <- toupper(model)
    for (j in 1:length(scenarios)) {
      scenario = scenarios[j]
      output[i, j+1] <- round(mean(df_round %>% filter(Scenario==scenario & Model==toupper(model) & Dataset==named_data[dataset]) %>% pull(AUC)), 2)
      output[i+length(datasets)*length(models), j+1] <- round(mean(df_round %>% filter(Scenario==scenario & Model==toupper(model) & Dataset==named_data[dataset]) %>% pull(SD)), 2)
      output[i+2*length(datasets)*length(models), j+1] <- formatC(round(mean(df_round %>% filter(Scenario==scenario & Model==toupper(model) & Dataset==named_data[dataset]) %>% pull(Training.Time)), 0), format='d', big.mark=',')
      output[i+3*length(datasets)*length(models), j+1] <- round(mean(df_round %>% filter(Scenario==scenario & Model==toupper(model) & Dataset==named_data[dataset]) %>% pull(Testing.Time)), 1)
    }
    i <- i + 1
  }
}
print(xtable(output), include.rownames=FALSE)

i <- 1
output <- matrix(nrow=3*length(datasets)*length(models), ncol=1+length(scenarios))
for (dataset in datasets) {
  for (model in models) {
    output[i, 1] <- toupper(model)
    output[i+length(datasets)*length(models), 1] <- toupper(model)
    output[i+2*length(datasets)*length(models), 1] <- toupper(model)
    for (j in 1:length(scenarios)) {
      scenario = scenarios[j]
      output[i, j+1] <- sprintf('%1.1f%%', mean(df_round_wo_online %>% filter(Scenario==scenario & Model==toupper(model) & Dataset==named_data[dataset]) %>% pull(AUC.Imp))*100)
      output[i+length(datasets)*length(models), j+1] <- round(mean(df_round_wo_online %>% filter(Scenario==scenario & Model==toupper(model) & Dataset==named_data[dataset]) %>% pull(Retrain.Count)), 0)
      output[i+2*length(datasets)*length(models), j+1] <- round(mean(na.omit(df_round_wo_online %>% filter(Scenario==scenario & Model==toupper(model) & Dataset==named_data[dataset]) %>% pull(EC))), 2)
    }
    i <- i + 1
  }
}
print(xtable(output), include.rownames=FALSE)


scenarios <- c('HT', 'ARF', 'AUE')
output <- matrix(nrow=4*length(datasets), ncol=length(scenarios))
i <- 1
for (dataset in datasets) {
  for (j in 1:length(scenarios)) {
    scenario = scenarios[j]
    output[i, j] <- round(mean(df_round %>% filter(Scenario==scenario & Dataset==named_data[dataset]) %>% pull(AUC)), 2)
    output[i+length(datasets), j] <- round(mean(df_round %>% filter(Scenario==scenario & Dataset==named_data[dataset]) %>% pull(SD)), 2)
    output[i+2*length(datasets), j] <- formatC(round(mean(df_round %>% filter(Scenario==scenario & Dataset==named_data[dataset]) %>% pull(Training.Time)), 0), format='d', big.mark=',')
    output[i+3*length(datasets), j] <- round(mean(df_round %>% filter(Scenario==scenario & Dataset==named_data[dataset]) %>% pull(Testing.Time)), 1)
  }
  i <- i + 1
}
print(xtable(output), include.rownames=FALSE)
