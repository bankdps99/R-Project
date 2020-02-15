install.packages("lattice")
install.packages("GGally")
install.packages("rattle")

library(rpart)
library(rpart.plot)
library(plyr)
library(tidyverse)
library(caret)
library(GGally)
library(stringr)
library(rattle)
library(pROC)
library(ROCR)
library(skimr)

set.seed(999)

fraud_raw <- read_csv("Fraud_raw.csv")

#Data Exploration
#skim(fraud_raw)
#glimpse(fraud_raw)

#Identify letter-prefix under nameOrig and nameDest variables which maybe useful for data interpretation
fraud_df <- fraud_raw %>%
mutate(name_orig_first = str_sub(nameOrig,1,1)) %>%
mutate(name_dest_first = str_sub(nameDest,1,1)) %>%
select(-nameOrig, -nameDest)

#Extract unique prefixes in nameDest
unique(fraud_df$name_dest_first)
#Convert unique prefixes to factor
fraud_df$name_dest_first <- as.factor(fraud_df$name_dest_first)
#Show amount of unique prefixes in table
table(fraud_df$name_dest_first)

#Unique prefixes in nameOrig
unique(fraud_df$name_orig_first)

#Only one prefix under nameOrig with no significance
#Drop nameOrig and isFlaggedFraud
fraud_df2 <- fraud_df %>%
select(-name_orig_first, -isFlaggedFraud) %>%
#To let isFraud,type and step show up in order following by others (to easier check isFraud number later)
select(isFraud, type, step, everything()) 


#glimpse(fraud_df2)

#Convert char to factor
fraud_df2$type <- as.factor(fraud_df2$type)
fraud_df2$isFraud <- as.factor(fraud_df2$isFraud)

#glimpse(fraud_df2)

#Recode numerical values of 0 and 1 to 'No' and 'Yes' for modelling purpose
fraud_df2$isFraud <- recode_factor(fraud_df2$isFraud, `0` = "No", `1` = "Yes")

summary(fraud_df2)
##There are only 8213 records for isFraud cases

#Check data only fraud case and get all fraud transactions from dataset
fraud_only <- fraud_df2 %>%
filter(isFraud == "Yes") 
summary(fraud_only)

#Sum shows fraud happening under cash_out and transfer types equally > filter
#Amount max = 10MM

#Drop insignificant variables
fraud_df3 <- fraud_df2 %>%
filter(type %in% c("CASH_OUT", "TRANSFER")) %>%
filter(name_dest_first == "C") %>%
#Drop level in variable by filtering out the amount of more than 10M
#No transaction above 10M, filter out 
filter(amount <= 10000000) %>%
select(-name_dest_first)

summary(fraud_df3)

###############################
#Handling imbalanced dataset 
##############################
#Create undersampling dataset
##############################

#Viszualize imbalanced data 
ggplot(fraud_df3, aes(fill = isFraud)) + stat_count(mapping = aes(x = `isFraud`))

#Plan to create 50:50 for not fraud and fraud dataset
##For not fraud dataset (not_fraud = fraud = 8213)
not_fraud <- fraud_df3 %>%
filter(isFraud == "No") %>%
sample_n(8213)

##For fraud dataset
is_fraud <- fraud_df3 %>%
filter(isFraud == "Yes")

#Row-Combination not-fruad and fraud dataset per step
full_sample <- rbind(not_fraud, is_fraud) %>%
#Arrangment following steps as it indicates the hour within the month that this data was captured
arrange(step)

summary(full_sample)

#Check pattern fraud by amount
ggplot(full_sample, aes(type, amount, color = isFraud)) +geom_point(alpha = 0.01) + geom_jitter()                                    

#Perform correlation test to evaluate the association between variables.
ggpairs(full_sample)

#Pre-processing the full dataset for modeling process
preproc_model <- preProcess(fraud_df3[, -1],method = c("center", "scale", "nzv"))
fraud_preproc <- predict(preproc_model, newdata = fraud_df3[, -1])
#Combind the results to the pre-processed data
fraud_and_result <- cbind(isFraud = fraud_df3$isFraud, fraud_preproc)

summary(fraud_and_result)

#Finalize dataframe
model_df <- fraud_and_result
is_fraud <- model_df %>%
filter(isFraud == "Yes")

not_fraud <- model_df %>%
filter(isFraud == "No") %>%
sample_n(8213)

# To mix up the sample set by step again
model_full_sample <- rbind(is_fraud, not_fraud) %>%
arrange(step)

summary(model_full_sample)

#Split model to train and test
in_train <- createDataPartition(y = model_full_sample$isFraud, p = .70, list = FALSE)
train <- full_sample[in_train, ] 
test <- full_sample[-in_train, ] 

###############################
####Random Forest model########
###############################

control <- trainControl(method='repeatedcv', number=10, repeats=3, search='grid')

#First code set for automatically select mtry producing highest accuracy
#tunegrid
#1.1 grid <- expand.grid(.mtry=c(1:5))

#1.1 Training with auto selecting highest accuracy, best mtry
#rf_model <- train(isFraud ~ ., data = train,method = 'rf',metric = "Accuracy",tuneGrid = grid)
#print(rf_model)
#plot(rf_model)

#1.1 Predict on test set
#rf_test_pred <- predict(rf_model, test)
#confusionMatrix(test$isFraud, rf_test_pred, positive = "Yes")

#Assign manual mtry for experimenting various mtry
#1.2 
grid <- expand.grid(.mtry = 1, .ntree = seq(25, 150, by = 25))

#1.2 Training
rf_model <- train(isFraud ~ ., data = train, method="rf", metric = "Accuracy", TuneGrid = grid, trControl=control)
print(rf_model$finalModel)
plot(rf_model$finalModel)

#1.2 Predict on test set
rf_test_pred <- predict(rf_model, test)
confusionMatrix(test$isFraud, rf_test_pred, positive = "Yes")

#Showing variable importance in RF technique
installed.packages("randomForest")
library(randomForest)
varImpPlot(rf_model$finalModel)

#ROC Curve
rf_probs <- predict(rf_model, test, type = "prob")
rf_ROC <- roc(response = test$isFraud, predictor = rf_probs$Yes, levels = levels(test$isFraud))

plot(rf_ROC, col = "red")
auc(rf_ROC)


#Take data sample for submission

not_fraud_submission <- fraud_df3 %>%
  filter(isFraud == "No") %>%
  sample_n(100)

##For fraud dataset
is_fraud_Submission <- fraud_df3 %>%
  filter(isFraud == "Yes") %>%
  sample_n(100)

#Row-Combination not-fruad and fraud dataset per step
full_sample_Submission <- rbind(not_fraud_submission, is_fraud_Submission) %>%
  #Arrangment following steps as it indicates the hour within the month that this data was captured
  arrange(step)

#summary(full_sample_Submission)

write.csv(full_sample_Submission, file = "SampleFile_Submission.csv")
