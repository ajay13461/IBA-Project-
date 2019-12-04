## Importing Packages
# Load all the packages required for the analysis
library(dplyr) # Data Manipulation
#install.packages("Amelia")
library(Amelia) # Missing Data: Missings Map
library(ggplot2) # Visualization
library(scales) # Visualization
library(caTools) # Prediction: Splitting Data
#install.packages("car")
library(car) # Prediction: Checking Multicollinearity
#install.packages("ROCR")
library(ROCR) # Prediction: ROC Curve
library(e1071) # Prediction: SVM, Naive Bayes, Parameter Tuning
library(rpart) # Prediction: Decision Tree
#install.packages("rpart.plot")
library(rpart.plot) # Prediction: Decision Tree
library(randomForest) # Prediction: Random Forest
library(caret) # Prediction: k-Fold Cross Validation

#Set Working Directory
setwd("C:\\Users\\Acer\\Desktop\\IBA Project")

## Reading the data
train = read.csv("train.csv",
                 na.strings = c("","NA","Na","na"))
sub = read.csv("gender_submission.csv",
                 na.strings = c("","NA","Na","na"))
test1 = read.csv("test.csv",
                 na.strings = c("","NA","Na","na"))
#Merging test & gender_submission
test = merge(test1, sub, by = "PassengerId")
## Checking Structure & Summary (this will also help to find out the missing values)
str(train)
str(test)
summary(train)
summary(test)

## Feature Engineering

# Missing Value Impustation
df = rbind(train,test)
train$Age[is.na(train$Age)]= mean(df$Age,na.rm=T)
test$Age[is.na(test$Age)]= mean(df$Age,na.rm=T)
train = subset(train, !is.na(Embarked))
test$Fare[is.na(test$Fare)] = mean(df$Fare,na.rm=T)

# Title Feature extraction
names_train <- train$Name
title_train <-  gsub("^.*, (.*?)\\..*$", "\\1", names_train)
train$title <- title_train
table(train$title)
names_test <- test$Name
title_test <-  gsub("^.*, (.*?)\\..*$", "\\1", names_test)
test$title <- title_test
table(test$title)
train$title[train$title == 'Mlle'] <- 'Miss' 
train$title[train$title == 'Ms'] <- 'Miss' 
train$title[train$title == 'Mme'] <- 'Mrs' 
train$title[train$title == 'Lady'] <- 'Miss'
train$title[train$title == 'Dona'] <- 'Miss' 
train$title[train$title == 'Capt'] <- 'Officer' 
train$title[train$title == 'Col'] <- 'Officer' 
train$title[train$title == 'Major'] <- 'Officer' 
train$title[train$title == 'Dr'] <- 'Officer' 
train$title[train$title == 'Rev'] <- 'Officer' 
train$title[train$title == 'Don'] <- 'Officer' 
train$title[train$title == 'Sir'] <- 'Officer' 
train$title[train$title == 'the Countess'] <- 'Officer' 
train$title[train$title == 'Jonkheer'] <- 'Officer'
test$title[test$title == 'Mlle'] <- 'Miss' 
test$title[test$title == 'Ms'] <- 'Miss' 
test$title[test$title == 'Mme'] <- 'Mrs' 
test$title[test$title == 'Lady'] <- 'Miss'
test$title[test$title == 'Dona'] <- 'Miss' 
test$title[test$title == 'Capt'] <- 'Officer' 
test$title[test$title == 'Col'] <- 'Officer' 
test$title[test$title == 'Major'] <- 'Officer' 
test$title[test$title == 'Dr'] <- 'Officer' 
test$title[test$title == 'Rev'] <- 'Officer' 
test$title[test$title == 'Don'] <- 'Officer' 
test$title[test$title == 'Sir'] <- 'Officer' 
test$title[test$title == 'the Countess'] <- 'Officer' 
test$title[test$title == 'Jonkheer'] <- 'Officer'

# Removal of Redundant Columns or Variables
train$Cabin = NULL
test$Cabin = NULL
train$PassengerId = NULL
test$PassengerId = NULL
train$Ticket = NULL
test$Ticket = NULL
train$Name = NULL
test$Name = NULL

# Conversion of variables into proper variable type
cols1=c("Survived","Pclass", "title")
for (i in cols1){
  train[,i]=as.factor(train[,i])
}
cols2=c("Survived","Pclass", "title")
for (i in cols2){
  test[,i]=as.factor(test[,i])
}

# lets prepare and keep data in the proper format
train = train[c("Pclass","Sex","Embarked","title","Age","SibSp","Parch","Fare","Survived")]
test = test[c("Pclass","Sex","Embarked","title","Age","SibSp","Parch","Fare","Survived")]

# Summary & Structure of Cleaned Data
summary(train)
summary(test)
str(train)
str(test)

## EDA & Feature Engineering
 
#Visualize P class which is the best proxy for Rich and Poor
ggplot(train[1:889,],aes(x = Pclass,fill=factor(Survived))) +
  geom_bar() +
  ggtitle("Pclass v/s Survival Rate")+
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived")#Inference: Rich passenger survival rate is much better than poor passenger.

#Visualize the 3-way relationship of sex, pclass, and survival
ggplot(train[1:889,], aes(x = Sex, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) + 
  ggtitle("3D view of sex, pclass, and survival") +
  xlab("Sex") +
  ylab("Total Count") +
  labs(fill = "Survived")#Inference: In the all the class female Survival rate is better than Men

# Which title has better survival rate
ggplot(train[1:889,],aes(x = title,fill=factor(Survived))) +
  geom_bar() +
  ggtitle("Title V/S Survival rate")+
  xlab("Title") +
  ylab("Total Count") +
  labs(fill = "Survived")#Inference: Mr. has better survival rate; Mrs & Miss has bad survival rate

#3-way of relationship of Title, Pclass, and Survival
ggplot(train[1:889,], aes(x = title, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) + 
  ggtitle("3-way relationship of Title, Pclass, and Survival") +
  xlab("Title") +
  ylab("Total Count") +
  labs(fill = "Survived")

#is there any association between Survial rate and where he get into the Ship.   
ggplot(train[1:889,],aes(x = Embarked,fill=factor(Survived))) +
  geom_bar() +
  ggtitle("Embarked vs Survival") +
  xlab("Embarked") +
  ylab("Total Count") +
  labs(fill = "Survived")

#Lets further divide the grpah by Pclass
ggplot(train[1:889,], aes(x = Embarked, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) + 
  ggtitle("Pclass vs Embarked vs survival") +
  xlab("Embarked") +
  ylab("Total Count") +
  labs(fill = "Survived")

### Modelling

#Logistic Regression
summary(train)
# Show the correlation of numeric features
cor(train[,unlist(lapply(train,is.numeric))])#Multicolinearity among continuous variable doesn't exist


#Chi Square Test
ps = chisq.test(train$Pclass, train$Sex)$p.value
pe = chisq.test(train$Pclass, train$Embarked)$p.value
pt = chisq.test(train$Pclass, train$title)$p.value
se = chisq.test(train$Sex, train$Embarked)$p.value
st = chisq.test(train$Sex, train$title)$p.value
et = chisq.test(train$Embarked, train$title)$p.value
cormatrix = matrix(c(0, ps, pe, pt,
                     ps, 0, se, st,
                     pe, se, 0, et,
                     pt, st, et, 0), 
                   4, 4, byrow = TRUE)
row.names(cormatrix) = colnames(cormatrix) = c("Pclass", "Sex", "Embarked", "title")
cormatrix # All the p values are <0.5 so there might be multicolinearity
log.reg = glm(Survived ~ ., family = binomial(link='logit'), data = train)
summary(log.reg)
#Sex has very low significance though before dropping let us check VIF
vif(log.reg)
#Sex has very hih GVIF score = 3233.471. So it is casing multicolinearity with title. We will drop Sex.
log.reg2 = glm(Survived ~ .-Sex, family = binomial(link='logit'), data = train)
summary(log.reg2)
vif(log.reg2)
durbinWatsonTest(log.reg)
#all the vif values are below close to 1 which is desirable
#DW Statstic value is 1.963 which signifies no autocorreleation.
#All the continuous variables are significant.
#Embarked has very low significance. but dropping more variables will mean losing out the information.
#So we will keep the model and calculate evaluation scores.
prob_pred = predict(log.reg2, type = 'response', newdata = train[-9])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
cm=table(train$Survived,y_pred)
confusionMatrix(cm)

#Evaluation Scores as per the Kaggle Copetition requirement.
log.reg.score=confusionMatrix(cm)

##Random Forest
set.seed(1234)
rf.1 = randomForest(x = train[-9],
                          y = train$Survived,
                          ntree = 500)
rf.1
varImpPlot(rf.1)
y_pred=predict(rf.1, newdata = train[-9])
confusionMatrix(y_pred,train$Survived)

set.seed(1234)
rf.2 = randomForest(x = train[c("Pclass","Sex","title","Age","Fare","SibSp")],
                    y = train$Survived,
                    ntree = 500)
head(train)
rf.2
varImpPlot(rf.2)
y_pred=predict(rf.2, newdata = test)
confusionMatrix(y_pred,test$Survived)
#Cross Validation
set.seed(2348)
cv10_1 <- createMultiFolds(train[,9], k = 10, times = 10)

# Set up caret's trainControl object per above.
ctrl_1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv10_1)



set.seed(1234)
rf.4<- train(x = train[c("Pclass","Sex","title","Age","Fare","SibSp")],
             y = train[,9], method = "rf", tuneLength = 3,
             ntree = 500, trControl =ctrl_1)
rf.4

#Decision Tree

# Fitting Decision Tree Classification Model to the Training set
classifier = rpart(Survived ~ ., data = train, method = 'class')
# Tree Visualization
rpart.plot(classifier, extra=4)

# Predicting the Validation set results
y_pred = predict(classifier, newdata = test[,-which(names(test)=="Survived")], type='class')

# Checking the prediction accuracy
table(test$Survived, y_pred) # Confusion matrix
error <- mean(test$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

## Accuracy of a single tree is 0.933. Overfitting can easily occur in Decision Tree classification. We can idenfity that evaluating the model using k-Fold Cross Validation. Or we might be able to improve the model. Let's do 10-fold cross validation to find out whether we could improve the model.
# Applying k-Fold Cross Validation
set.seed(789)
folds = createMultiFolds(train$Survived, k = 10, times = 5)
control <- trainControl(method = "repeatedcv", index = folds)
classifier_cv <- train(Survived ~ ., data = train, method = "rpart", trControl = control)

# Tree Visualization
rpart.plot(classifier_cv$finalModel, extra=4)

# Predicting the Validation set results
y_pred = predict(classifier_cv, newdata = test[,-which(names(test)=="Survived")])

# Checking the prediction accuracy
table(test$Survived, y_pred) # Confusion matrix

error <- mean(test$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

## We were able to improve the model after 10-fold cross validation. 
## The accuracy has been improved to 0.9354 but note the improved model uses only three features Title, Pclass and Fare for classification.


#Random Forest

# Fitting Random Forest Classification to the Training set
set.seed(432)
classifier = randomForest(Survived ~ ., data = train)

# Choosing the number of trees
plot(classifier)

## The green, black and red lines represent error rate for death, overall and survival, respectively. The overall error rate converges to around 17%. 
## Interestingly, our model predicts death better than survival. 
## Since the overall error rate converges to a constant and does not seem to further decrease, our choice of default 500 trees in the randomForest function is a good choice.

# Predicting the Validation set results
y_pred = predict(classifier, newdata = test[,-which(names(test)=="Survived")])

# Checking the prediction accuracy
table(test$Survived, y_pred) # Confusion matrix

error <- mean(test$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

# Accuracy is 0.9019.

# Applying k-Fold Cross Validation
set.seed(651)
folds = createMultiFolds(train$Survived, k = 10)
control <- trainControl(method = "repeatedcv", index = folds)
classifier_cv <- train(Survived ~ ., data = train, method = "rf", trControl = control)

# Predicting the Validation set results
y_pred = predict(classifier_cv, newdata = test[,-which(names(test)=="Survived")])

# Checking the prediction accuracy
table(test$Survived, y_pred) # Confusion matrix

error <- mean(test$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

## Accuracy went down to 0.8421. We were not able to improve the random forest model using 10-fold cross validation.
## The random Forest classification suffers in terms of interpretability. We are unable to visualize the 500 trees and identify important features of the model. 
## However, we can assess the Feature Importance using the Gini index measure. Let's plot mean Gini index across all trees and identify important features.

# Feature Importance
gini = as.data.frame(importance(classifier))
gini = data.frame(Feature = row.names(gini), 
                  MeanGini = round(gini[ ,'MeanDecreaseGini'], 2))
gini = gini[order(-gini[,"MeanGini"]),]

ggplot(gini,aes(reorder(Feature,MeanGini), MeanGini, group=1)) + 
  geom_point(color='red',shape=17,size=2) + 
  geom_line(color='blue',size=1) +
  scale_y_continuous(breaks=seq(0,60,10)) +
  xlab("Feature") + 
  ggtitle("Mean Gini Index of Features") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## The feature Title has the highest mean gini index, hence the highest importance. Fare is also realtively high important and it is followed by Age of the passengers.

## Comparison of models

#Logistics Regression
## Accuracy  = 0.8279

#Decison Tree
## Accuracy  = 0.933

#Decision Tree with 10-fold cross-validation
## Accuracy  = 0.9354


#Random Forest
## Accuracy  = 0.9019

#Random Forest with 10-fold cross-validation
## Accuracy  = 0.8421
