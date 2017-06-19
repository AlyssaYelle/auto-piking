library(genlasso)
library(foreach)
library(parallel)
library(doMC)
        
setwd('~/Projects/auto-piking')
transect.name = "ASB:JKB2d:F50T01a"
X = read.table(sprintf('data/%s_srf_unfoc_x.txt', transect.name), header=F)
y = scan(sprintf('data/%s_srf_unfoc_y.txt', transect.name))
# yhat = integer(length(y))
# dim(X)[1]
doMC::registerDoMC(parallel::detectCores())
yhat = foreach(i=1:dim(X)[1]) %dopar% {
    x = as.numeric(X[i,])
    a = trendfilter(x, ord=1)
    cv = cv.trendfilter(a)
    xsmooth = coef(a, lambda=cv$lambda.1se)$beta
    print(c(i, which.max(xsmooth) - 1, y[i]))
    which.max(xsmooth) - 1
}
yhat = as.integer(yhat)

plot(y, yhat)
abline(0,1)

# Just use the raw pixel intensity and take the max
for (transect.name in gsub(".png", "", list.files(path="screenshots/focused/bed/picked"))){
    X = read.table(sprintf('data/%s_srf_unfoc_x.txt', transect.name), header=F)
    y = scan(sprintf('data/%s_srf_unfoc_y.txt', transect.name))
    zhat = foreach(i=1:dim(X)[1]) %do% {
    x = as.numeric(X[i,])
    which.max(x) - 1
    }
    zhat = as.integer(zhat)
    plot(y, zhat)
    abline(0,1)

}


