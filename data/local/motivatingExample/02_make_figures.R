
### Make plots for solar intervention example on temperature:
## ---------------------------------------------------

# Sebastian Sippel
# 15.10.2021

# y and yhat are radiative CO2 forcing in W/m^2. following: ∆RF = 5.22ln(C/Co) based on https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/98GL01908

## PLOT DATASET OVERVIEW:
setwd("/net/h2o/climphys1/sippels/_projects/solar_intervention//")
overview_NOIV = read.table(file = "00_overview_NOIV.txt", sep=";", header = T)
overview_trainIV = read.table(file = "00_overview_trainIV.txt", sep=";", header = T)
overview_testIV = read.table(file = "00_overview_testIV.txt", sep=";", header = T)

pred_cntl = read.table(file = "01_pred_cntl.txt", sep=";", header = T)
pred_coldsun = read.table(file = "01_pred_coldsun.txt", sep=";", header = T)
pred_warmsun = read.table(file = "01_pred_warmsun.txt", sep=";", header = T)


# i. Plot dataset overview: 
setwd("figures_v2/")

pdf("_01_overview_solar_intervention.pdf", width=12, height=4)
par(mfrow=c(1, 3), mar=c(4,4,3,1))
{
ylim_dataoverview = c(-5, 10)
ylab_dataoverview = c("Global Mean Temperature Anomaly [°C]")
flim = c(-5, 15)

## a. No Interventions:
plot(x = overview_NOIV$year[which(overview_NOIV$scen == "cntl")], overview_NOIV$T_glm[which(overview_NOIV$scen == "cntl")], type="n", ylim = ylim_dataoverview, xlim = c(500, 2500),
     xlab = "Year", ylab = ylab_dataoverview, main="No Interventions", xaxt="n")
axis(side = 1, at = seq(500, 5000, 500), tick = T, labels=T)

lines(x = overview_NOIV$year[which(overview_NOIV$scen == "cntl")], 
      y = overview_NOIV$T_glm[which(overview_NOIV$scen == "cntl")], col = "darkorange")

# 1%CO2 in cntl
start.year.un = unique(overview_NOIV$start.year[which(overview_NOIV$scen == "1perCO2")])
sapply(X = 1:length(start.year.un), FUN=function(i) { ix = which(overview_NOIV$scen == "1perCO2" & overview_NOIV$start.year == start.year.un[i]); 
lines(x = overview_NOIV$year[ix], overview_NOIV$T_glm[ix], col = "gray40") })

legend("top", c("PI-Control", "1%/yr CO2 increase"), lwd = 2, col = c("darkorange", "gray40"), inset = 0.02, cex = 0.9)


## b. TRAINING DATA: 
plot(x = overview_NOIV$year[which(overview_NOIV$scen == "cntl")], overview_NOIV$T_glm[which(overview_NOIV$scen == "cntl")], type="n", ylim = ylim_dataoverview, xlim = c(500, 2500),
     xlab = "", ylab = ylab_dataoverview, main="Small Interventions (Training Data)", xaxt="n")
axis(side = 1, at = seq(500, 5000, 500), tick = T, labels=F)

lines(x = 501:2500, y = overview_trainIV$T_glm[which(overview_trainIV$scen == "cntl" & overview_trainIV$year %in% c(501:2500))], col = "darkorange")
lines(x = 1001:2000, y = overview_trainIV$T_glm[which(overview_trainIV$scen == "plus6W" & overview_trainIV$year %in% c(1001:2000))], col = "red")
lines(x = 1001:2000, y = overview_trainIV$T_glm[which(overview_trainIV$scen == "minus6W" & overview_trainIV$year %in% c(1001:2000))], col = "lightblue")

# 1%CO2 in cntl
start.year.un = c(1000, 1100) 
sapply(X = 1:length(start.year.un), FUN=function(i) { ix = which(overview_trainIV$scen == "1perCO2" & overview_trainIV$start.year == start.year.un[i]); 
lines(x = overview_trainIV$year[ix], overview_trainIV$T_glm[ix], col = "gray40") })

# 1%CO2 in plus
start.year.un = c(1040, 1900, 1950, 2000) 
sapply(X = 1:length(start.year.un), FUN=function(i) { ix = which(overview_trainIV$scen == "plus6W.1perCO2" & overview_trainIV$start.year == start.year.un[i]); 
lines(x = overview_trainIV$year[ix], y = overview_trainIV$T_glm[ix], col = "gray40") })

# 1%CO2 in minus25
start.year.un = c(1400, 1900, 1950, 2000) 
sapply(X = 1:length(start.year.un), FUN=function(i) { ix = which(overview_trainIV$scen == "minus6W.1perCO2" & overview_trainIV$start.year == start.year.un[i]); 
lines(x = overview_trainIV$year[ix], overview_trainIV$T_glm[ix], col = "gray40") })

legend("top", c("warm sun* (~ +6 W m-2)", "cool sun* (~ -6 W m-2)"), lwd = 2, col = c("red", "lightblue"), inset = 0.02, cex = 0.9)


## c. FULL DATASET / STRONG INTERVENTIONS: 
plot(x = overview_trainIV$year[which(overview_trainIV$M$scen == "cntl")], overview_trainIV$TREFHT_glm[which(overview_trainIV$scen == "cntl")], type="n", ylim = ylim_dataoverview, xlim = c(500, 2500),
     xlab = "Year", ylab = ylab_dataoverview, main="Strong Interventions (Test Data)")

lines(x = overview_testIV$year[which(overview_testIV$scen == "cntl")], overview_testIV$T_glm[which(overview_testIV$scen == "cntl")], col = "darkorange")
lines(x = overview_testIV$year[which(overview_testIV$scen == "minus25W")], overview_testIV$T_glm[which(overview_testIV$scen == "minus25W")], col = "darkblue")
lines(x = overview_testIV$year[which(overview_testIV$scen == "plus25W")], overview_testIV$T_glm[which(overview_testIV$scen == "plus25W")], col = "darkred")

# lines(x = 1001:2000, y = mv.all$Y$TREFHT_glm[which(mv.all$M$scen == "cntl" & mv.all$M$year %in% c(1001:2000))] - 273.15 + c(delta_plus25W %*% areaw) / scale.div, col = make.transparent.color("red", alpha = 100))
# lines(x = 1001:2000, y = mv.all$Y$TREFHT_glm[which(mv.all$M$scen == "cntl" & mv.all$M$year %in% c(2001:3000))] - 273.15 + c(delta_minus25W %*% areaw) / scale.div, col = make.transparent.color("lightblue", alpha = 100))

# plot 1%CO2 runs for test data: 
start.year.un = unique(overview_testIV$start.year[which(overview_testIV$scen == "1perCO2")])
sapply(X = 1:length(start.year.un), FUN=function(i) { ix = which(overview_testIV$scen == "1perCO2" & overview_testIV$start.year == start.year.un[i]); 
lines(x = overview_testIV$year[ix], overview_testIV$T_glm[ix], col = "gray40") })

start.year.un = unique(overview_testIV$start.year[which(overview_testIV$scen == "plus25W.1perCO2")])
sapply(X = 1:length(start.year.un), FUN=function(i) {ix = which(overview_testIV$scen == "plus25W.1perCO2" & overview_testIV$start.year == start.year.un[i]); 
lines(x = overview_testIV$year[ix], overview_testIV$T_glm[ix], col = "gray40") })

start.year.un = unique(overview_testIV$start.year[which(overview_testIV$scen == "minus25W.1perCO2")])
sapply(X = 1:length(start.year.un), FUN=function(i) {ix = which(overview_testIV$scen == "minus25W.1perCO2" & overview_testIV$start.year == start.year.un[i]); 
lines(x = overview_testIV$year[ix], overview_testIV$T_glm[ix], col = "gray40") })

legend("top", c("hot sun (+25 W m-2)", "cold sun (+25 W m-2)"), lwd = 2, col = c("darkred", "darkblue"), inset = 0.02, ncol = 2, cex = 0.9)

dev.off()
}




# ii. Plot predictions and residuals:
pdf("_02_anchor_solar_intervention.pdf", width=12, height=12)
par(mfrow=c(1, 3), mar=c(4,4,3,1))
{
  library(hydroGOF)
  flim = c(-5, 15)
  
## NO INTERVENTIONS:
par(mfrow=c(3,3), mar=c(4,4,1,1))
plot(x = pred_cntl$yhat_cntl_A0, y = pred_cntl$y_cntl, main = "No Interventions", pch = 16, col = "darkorange", 
     xlab = "Predicted CO2 Forcing [W m-2]", ylab = "True CO2 Forcing  [W m-2]", xlim = flim, ylim = flim)
abline(0, 1, col="darkgray", lwd = 2)
legend("topleft", c(paste("RMSE = ", round(rmse(pred_cntl$yhat_cntl_A0, pred_cntl$y_cntl), 2), sep=""), 
                    paste("r = ", round(cor(pred_cntl$yhat_cntl_A0, pred_cntl$y_cntl), 2), sep="")), title = expression(paste(gamma, " = 0")))

plot(x = pred_cntl$yhat_cntl_A1, y = pred_cntl$y_cntl, main = "No Interventions", pch = 16, col = "darkorange", 
     xlab = "Predicted CO2 Forcing [W m-2]", ylab = "True CO2 Forcing  [W m-2]", xlim = flim, ylim = flim)
abline(0, 1, col="darkgray", lwd = 2)
legend("topleft", c(paste("RMSE = ", round(rmse(pred_cntl$yhat_cntl_A1, pred_cntl$y_cntl), 2), sep=""), 
                    paste("r = ", round(cor(pred_cntl$yhat_cntl_A1, pred_cntl$y_cntl), 2), sep="")), title = expression(paste(gamma, " = 1")))

plot(x = pred_cntl$yhat_cntl_A100, y = pred_cntl$y_cntl, main = "No Interventions", pch = 16, col = "darkorange", 
     xlab = "Predicted CO2 Forcing [W m-2]", ylab = "True CO2 Forcing  [W m-2]", xlim = flim, ylim = flim)
abline(0, 1, col="darkgray", lwd = 2)
legend("topleft", c(paste("RMSE = ", round(rmse(pred_cntl$yhat_cntl_A100, pred_cntl$y_cntl), 2), sep=""), 
                    paste("r = ", round(cor(pred_cntl$yhat_cntl_A100, pred_cntl$y_cntl), 2), sep="")), title = expression(paste(gamma, " = 100")))


## STRONG INTERVENTIONS:
plot(x = pred_cntl$yhat_cntl_A0, y = pred_cntl$y_cntl, main = "Strong Interventions", pch = 16, col = "darkorange", 
     xlab = "Predicted CO2 Forcing [W m-2]", ylab = "True CO2 Forcing  [W m-2]", xlim = flim, ylim = flim)
abline(0, 1, col="darkgray", lwd = 2)
points(x = pred_warmsun$yhat_warmsun_A0, y = pred_warmsun$y_warmsun, pch = 16, col = "darkred")
points(x = pred_coldsun$yhat_coldsun_A0, y = pred_coldsun$y_coldsun, pch = 16, col = "darkblue")
legend("topleft", c(paste("RMSE = ", round(rmse(c(pred_cntl$yhat_cntl_A0, pred_warmsun$yhat_warmsun_A0, pred_coldsun$yhat_coldsun_A0), 
                                                c(pred_cntl$y_cntl, pred_warmsun$y_warmsun, pred_coldsun$y_coldsun)), 2), sep=""), 
                    paste("r = ", round(cor(c(pred_cntl$yhat_cntl_A0, pred_warmsun$yhat_warmsun_A0, pred_coldsun$yhat_coldsun_A0), c(pred_cntl$y_cntl, pred_warmsun$y_warmsun, pred_coldsun$y_coldsun)), 2), sep="")), 
       title = expression(paste(gamma, " = 0")))


plot(x = pred_cntl$yhat_cntl_A1, y = pred_cntl$y_cntl, main = "Strong Interventions", pch = 16, col = "darkorange", 
     xlab = "Predicted CO2 Forcing [W m-2]", ylab = "True CO2 Forcing  [W m-2]", xlim = flim, ylim = flim)
abline(0, 1, col="darkgray", lwd = 2)
points(x = pred_warmsun$yhat_warmsun_A1, y = pred_warmsun$y_warmsun, pch = 16, col = "darkred")
points(x = pred_coldsun$yhat_coldsun_A1, y = pred_coldsun$y_coldsun, pch = 16, col = "darkblue")
legend("topleft", c(paste("RMSE = ", round(rmse(c(pred_cntl$yhat_cntl_A1, pred_warmsun$yhat_warmsun_A1, pred_coldsun$yhat_coldsun_A1), 
                                                c(pred_cntl$y_cntl, pred_warmsun$y_warmsun, pred_coldsun$y_coldsun)), 2), sep=""), 
                    paste("r = ", round(cor(c(pred_cntl$yhat_cntl_A1, pred_warmsun$yhat_warmsun_A1, pred_coldsun$yhat_coldsun_A1), c(pred_cntl$y_cntl, pred_warmsun$y_warmsun, pred_coldsun$y_coldsun)), 2), sep="")), 
       title = expression(paste(gamma, " = 1")))


plot(x = pred_cntl$yhat_cntl_A100, y = pred_cntl$y_cntl, main = "Strong Interventions", pch = 16, col = "darkorange", 
     xlab = "Predicted CO2 Forcing [W m-2]", ylab = "True CO2 Forcing  [W m-2]", xlim = flim, ylim = flim)
abline(0, 1, col="darkgray", lwd = 2)
points(x = pred_warmsun$yhat_warmsun_A100, y = pred_warmsun$y_warmsun, pch = 16, col = "darkred")
points(x = pred_coldsun$yhat_coldsun_A100, y = pred_coldsun$y_coldsun, pch = 16, col = "darkblue")
legend("topleft", c(paste("RMSE = ", round(rmse(c(pred_cntl$yhat_cntl_A100, pred_warmsun$yhat_warmsun_A100, pred_coldsun$yhat_coldsun_A100), 
                                                c(pred_cntl$y_cntl, pred_warmsun$y_warmsun, pred_coldsun$y_coldsun)), 2), sep=""), 
                    paste("r = ", round(cor(c(pred_cntl$yhat_cntl_A100, pred_warmsun$yhat_warmsun_A100, pred_coldsun$yhat_coldsun_A100), c(pred_cntl$y_cntl, pred_warmsun$y_warmsun, pred_coldsun$y_coldsun)), 2), sep="")), 
       title = expression(paste(gamma, " = 100")))



## STRONG INTERVENTIONS, residual correlations:
plot(x = pred_cntl$yhat_cntl_A0 - pred_cntl$y_cntl, y = rep(0, 641), main = "Strong Interventions, Residual correlation w. Anchor", pch = 16, col = "darkorange",
     xlab = "Prediction Residuals", ylab = "Solar Anchor [W/m2]", xlim = c(-6, 6), ylim = c(-30, 30))
points(x = pred_warmsun$yhat_warmsun_A0 - pred_warmsun$y_warmsun, y = rep(25, 1423), pch = 16, col = "darkred")
points(x = pred_coldsun$yhat_coldsun_A0 - pred_coldsun$y_coldsun, y = rep(-25, 1413), pch = 16, col = "darkblue")
legend("topleft", c(paste("r = ", round(cor(c(pred_cntl$yhat_cntl_A0 - pred_cntl$y_cntl, pred_warmsun$yhat_warmsun_A0 - pred_warmsun$y_warmsun, pred_coldsun$yhat_coldsun_A0 - pred_coldsun$y_coldsun), 
                                            c(rep(0, 641), rep(25, 1423), rep(-25, 1413))), 2), sep="")), 
       title = expression(paste(gamma, " = 0")))


plot(x = pred_cntl$yhat_cntl_A1 - pred_cntl$y_cntl, y = rep(0, 641), main = "Strong Interventions, Residual correlation w. Anchor", pch = 16, col = "darkorange",
     xlab = "Prediction Residuals", ylab = "Solar Anchor [W/m2]", xlim = c(-6, 6), ylim = c(-30, 30))
points(x = pred_warmsun$yhat_warmsun_A1 - pred_warmsun$y_warmsun, y = rep(25, 1423), pch = 16, col = "darkred")
points(x = pred_coldsun$yhat_coldsun_A1 - pred_coldsun$y_coldsun, y = rep(-25, 1413), pch = 16, col = "darkblue")
legend("topleft", c(paste("r = ", round(cor(c(pred_cntl$yhat_cntl_A1 - pred_cntl$y_cntl, pred_warmsun$yhat_warmsun_A1 - pred_warmsun$y_warmsun, pred_coldsun$yhat_coldsun_A1 - pred_coldsun$y_coldsun), 
                                            c(rep(0, 641), rep(25, 1423), rep(-25, 1413))), 2), sep="")), 
       title = expression(paste(gamma, " = 1")))


plot(x = pred_cntl$yhat_cntl_A100 - pred_cntl$y_cntl, y = rep(0, 641), main = "Strong Interventions, Residual correlation w. Anchor", pch = 16, col = "darkorange",
     xlab = "Prediction Residuals", ylab = "Solar Anchor [W/m2]", xlim = c(-6, 6), ylim = c(-30, 30))
points(x = pred_warmsun$yhat_warmsun_A100 - pred_warmsun$y_warmsun, y = rep(25, 1423), pch = 16, col = "darkred")
points(x = pred_coldsun$yhat_coldsun_A100 - pred_coldsun$y_coldsun, y = rep(-25, 1413), pch = 16, col = "darkblue")
legend("topleft", c(paste("r = ", round(cor(c(pred_cntl$yhat_cntl_A100 - pred_cntl$y_cntl, pred_warmsun$yhat_warmsun_A100 - pred_warmsun$y_warmsun, pred_coldsun$yhat_coldsun_A100 - pred_coldsun$y_coldsun), 
                                            c(rep(0, 641), rep(25, 1423), rep(-25, 1413))), 2), sep="")), 
       title = expression(paste(gamma, " = 100")))
dev.off()
}

