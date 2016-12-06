library(MASS)
library(dplyr)
library(lme4)
library(arm)
library(DAAG)
data = read.csv("/Users/LinXIAO/Desktop/841/final project/data.csv", header=T)
test = data[is.na(data$shot_made_flag),]
train = data[!is.na(data$shot_made_flag),]

# add home or guest
train[grepl("@", train$matchup), "home"] = 0
train[!grepl("@", train$matchup), "home"] = 1
train[,"minutes_left"] <- train$minutes_remaining+12*(train$period-1)
train = train[,!(names(train) %in% c("game_date", "matchup","minutes_remaining"))]
train$home = as.factor(train$home)
train$playoffs = as.factor(train$playoffs)
train$period = as.factor(train$period)

trainmade <- train[train$shot_made_flag==1,]
hist(trainmade$shot_distance, main="Made shots", xlab = "shot distance")
trainmiss <- train[train$shot_made_flag==0,]
hist(trainmiss$shot_distance, main="Missed shots", xlab = "shot distance")

attach(train)
###########################################################################################
# Q1: What is correlated to shot_made_flag?

for(i in 1:ncol(train)){
  summary(glm(train$shot_made_flag~train[,i], family = binomial("logit")))
}
ttttt = summarise(group_by(train, home), count = n(), made = sum(shot_made_flag))
summary(glm(cbind(ttttt$made, ttttt$count-ttttt$made)~c(0,1), family = binomial("logit")))

# playoff is independent, saying sth like this player is super professional; make xtable here


###########################################################################################
# shot_type and shot_zone range
# We still want to attack the rim since the shot_percentage is much higher for less than 8 feet
trainnew = train[,c("shot_zone_range", "shot_made_flag", "shot_type")]
table = summarise(group_by(trainnew, shot_zone_range, shot_type, shot_made_flag), count = n())
prop.table(xtabs(count~shot_type+shot_made_flag, table),1)

prop.table(xtabs(count~shot_zone_range+shot_made_flag, table),1)

table$shot_made_flag <- as.factor(table$shot_made_flag)
table$count <- as.double(table$count)
table.partial = xtabs(count~shot_type+shot_made_flag+shot_zone_range, table)

# conditional independent
mantelhaen.test(table.partial)

###########################################################################################

# Ordered logistic regression, refer to Kobe's performance
# Proportional-odds cumulative logit model
by_tailnum <- group_by(train, shot_zone_range, season)
delay <- print(summarise(by_tailnum, count = sum(shot_made_flag)),n=100)
delay$shot_zone_range <- factor(delay$shot_zone_range, levels = c("Less Than 8 ft.", "8-16 ft.", "16-24 ft.", "24+ ft.", "Back Court Shot"))
m.po = polr(shot_zone_range~season, weights = count, data = delay)
plot(exp(coef(m.po)), ylab = "exponential of estimates", xlab = "season")
###########################################################################################







###########################################################################################
# shot_distance and shot_type
## different intercept different slope is better model
m1 = glm(shot_made_flag~shot_distance,data = train, family = binomial("logit"))
m2 = glm(shot_made_flag~shot_distance+shot_type,data = train, family = binomial("logit"))
# indicates model does not fit well with lower than 0.05 pvalue
# hosmerlem(train$shot_made_flag, fitted(m2))
m3 = glm(shot_made_flag~shot_distance*shot_type, data = train,family = binomial("logit"))
anova(m2, m3, test="Chisq")
# no overdispersion effect
sum((residuals(m3, "pearson"))^2)/m3$df.residual
###########################################################################################


# what about home?

# shot distance and home
## different intercept different slope is better model
m1 = glm(shot_made_flag~shot_distance, data = train, family = binomial("logit"))
# common slope different intercept
m2 = glm(shot_made_flag~shot_distance+home, data = train, family = binomial("logit"))
# different intercept different slope
m3 = glm(shot_made_flag~shot_distance*home, data = train, family = binomial("logit"))
anova(m1, m3, test="Chisq")
anova(m2, m3, test="Chisq")
# common slope different intercept rules
sum((residuals(m3, "pearson"))^2)/m3$df.residual

###########################################################################################
# Generalized linear models
# dont see overdispersion: could be due to some reasons including underdispersion and overdispersion cancelling effect
m1 = glm(shot_made_flag~shot_distance, data = train, family = binomial("logit"))

step <- stepAIC(logit, direction="both")

logit = glm(shot_made_flag~shot_distance+factor(playoffs)+factor(period)+shot_zone_area+factor(shot_zone_range)+factor(home)+factor(season), 
          data = train, family = binomial("logit"))
probit = glm(shot_made_flag~shot_distance+factor(playoffs)+factor(period)+shot_zone_area+factor(shot_zone_range)+factor(home)+factor(season), 
          data = train, family = binomial("probit"))
cauchit = glm(shot_made_flag~shot_distance+factor(playoffs)+factor(period)+shot_zone_area+factor(shot_zone_range)+factor(home)+factor(season), 
          data = train, family = binomial("cauchit"))



# action type can replace combined shot type
p1 = glmer(shot_made_flag~shot_distance+(1|action_type), data = train, family = binomial("logit"))
p2 = glmer(shot_made_flag~shot_distance+(1|combined_shot_type), data = train, family = binomial("logit"))
p3 = glmer(shot_made_flag~shot_distance+(1|combined_shot_type)+(1|action_type), data = train, family = binomial("logit"))
p5 = glmer(shot_made_flag~shot_distance+(1|combined_shot_type/action_type), data = train, family = binomial("logit"))
p4 = glmer(shot_made_flag~shot_distance+(1+shot_distance|combined_shot_type/action_type), data = train, family = binomial("logit"))
p6 = glmer(shot_made_flag~shot_distance+(1|shot_type/action_type), data = train, family = binomial("logit"))
p9 = glmer(shot_made_flag~shot_distance+(1+shot_distance|combined_shot_type/shot_type), data = train, family = binomial("logit"))
p10 = glmer(shot_made_flag~shot_distance+(1+shot_distance|shot_type/action_type), data = train, family = binomial("logit"))

list = predict(p16, type = "response")

for(i in seq(0.35,0.6, length.out = 20)){
  pred = list>i
  print((dim(train)[1] - sum(abs(pred-train$shot_made_flag)))/dim(train)[1])
}

# Run on all
#train$playoffs = as.factor(train$playoffs)
#train$shot_distance = scale(train$shot_distance)
#train[,"minutes_left"] <- train$minutes_remaining+12*(train$period-1)
#train$minutes_left = scale(train$minutes_left)

sum((residuals(m3, "pearson"))^2)/m3$df.residual

p15 = glmer(shot_made_flag~shot_distance+(1|playoffs)+(1|period)+(1|shot_zone_area)+(1|shot_zone_range)+
              (1|home)+(1|season)+(1|combined_shot_type/action_type), data = train, family = binomial("logit"))
p16 = glmer(shot_made_flag~shot_distance+(1|playoffs)+(1|period)+(1|shot_zone_area)+(1|shot_zone_range)+
              (1|home)+(1|season)+(1|shot_type/action_type), data = train, family = binomial("logit"))

# Didnt converge for the following nested model
p17 = glmer(shot_made_flag~shot_distance+(1|period)+(1|shot_zone_area)+(1|shot_zone_range)+
              (1|home)+(1|season)+(1+shot_distance|shot_type/action_type), data = train, family = binomial("logit"))
p18 = glmer(shot_made_flag~shot_distance+(1|period)+(1|shot_zone_area)+(1|shot_zone_range)+
              (1|home)+(1|season)+(1+shot_distance|combined_shot_type/action_type), data = train, family = binomial("logit"))
# add another layer works better
p19 = glmer(shot_made_flag~shot_distance+(1|playoffs)+(1|period)+(1|shot_zone_area)+(1|shot_zone_range)+
              (1|home)+(1|season)+(1+shot_distance|shot_type/combined_shot_type/action_type), data = train, family = binomial("logit"))


########## consider add layer about shot zone
# run time too long

anova(p20, p21, test="Chisq")
# run anova
display(p41)

###########################################################################################

# bidirectional stepwise

###########################################################################################
# other topics:
# log odds ratio

# Professional player proven
# area effect? Denver
train1 = train
train1$opponent = as.character(train1$opponent)
train1[train1$opponent != "DEN","opponent"] <- "NOTDEN"
summarise(group_by(train1, opponent, home), count=n(), made = sum(shot_made_flag))

mat1 = matrix(c(712,319, 640+12500+11845, 300+5447+5399), byrow = T, ncol = 2)
summary(glm(cbind(mat1[,2], mat1[,1] - mat1[,2])~c(0,1), family = binomial("logit")))
fisher.test(mat1)

# Spurs?
# tend to shoot more middle range(16-24) compared to 0-8 and 8-16 when facing Spurs
train2 = train
train2$opponent = as.character(train2$opponent)
train2[train2$opponent != "SAS","opponent"] <- "NOTSAS"
ttt2 = summarise(group_by(train2, opponent, shot_zone_range), count=n(), made = sum(shot_made_flag))
mat2 = matrix(c(7315+5182, 6492, 542+398, 415), byrow = T, ncol = 2)
summary(glm(cbind(mat2[,2], mat2[,1])~c(0,1), family = binomial("logit")))



# right or left? Probably no different, indicating this guy is good at both side
summarise(group_by(train, shot_zone_area), count=n(), made = sum(shot_made_flag))
mat2 = matrix(c(3364+3132, 1215+1243, 3981+3859, 1523+1550), byrow = T, ncol = 2)
summary(glm(cbind(mat2[,2], mat2[,1] - mat2[,2])~c(0,1), family = binomial("logit")))

mat2 = matrix(c(3364, 1215, 3981, 1523), byrow = T, ncol = 2)
summary(glm(cbind(mat2[,2], mat2[,1] - mat2[,2])~c(0,1), family = binomial("logit")))


# center and non center are different
# He is shoting better in center
mat2 = matrix(c(11289, 5933, 3981+3859+3364+3132, 1215+1243+1523+1550), byrow = T, ncol = 2)
summary(glm(cbind(mat2[,2], mat2[,1] - mat2[,2])~c(0,1), family = binomial("logit")))

# So he takes more shots in center, especailly during playoffs, save play.
tmat2 = summarise(group_by(train, shot_zone_area, playoffs), count=n(), made = sum(shot_made_flag))
mat2 = matrix(c(sum(tmat2[c(6,8,10,12),"count"]), sum(tmat2[c(5,7,9,11),"count"]), 
                as.numeric(tmat2[4,"count"]), as.numeric(tmat2[3,"count"])), byrow = T, ncol = 2)
summary(glm(cbind(mat2[,2], mat2[,1])~c(0,1), family = binomial("logit")))

###########################################################################################

###  Bayesian??

###########################################################################################

###  diagnotics


###########################################################################################

###  predictions, compared to machine learning, on the test set (include LDA)


print(sessionInfo(), locale = FALSE)
