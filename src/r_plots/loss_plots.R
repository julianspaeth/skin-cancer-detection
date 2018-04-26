df1 <- read.table("loss_l1.txt", header=FALSE, sep = "\n")
x <- seq(1, 2000, by = 1)
x = x*500
smoothingSpline = smooth.spline(x, df1$V1, spar=0.5)
plot(x,df1$V1, type = "l", col = rgb(0, 0, 1, 0.3), xlab = "Iteration", ylab = "L1-Loss", ylim = c(0,4))
lines(smoothingSpline, col = "red", lwd = 3)
legend(c(500000, 800000), c(3.55, 4.0), legend=c("originale Daten", "geglättete Daten"),
       col=c("blue", "red"), lty=1:1, cex=1.0)


df2 <- read.table("loss_l2.txt", header=FALSE, sep = "\n")
smoothingSpline = smooth.spline(x, df2$V1, spar=0.3)
plot(x,df2$V1, type = "l", col = rgb(0, 0, 1, 0.3), xlab = "Iteration", ylab = "L2-Loss", ylim = c(0,4))
lines(smoothingSpline, col = "red", lwd = 3)
legend(c(500000, 800000), c(3.55, 4.0), legend=c("originale Daten", "geglättete Daten"),
       col=c("blue", "red"), lty=1:1, cex=1.0)


df3 <- read.table("loss_ce.txt", header=FALSE, sep = "\n")
smoothingSpline = smooth.spline(x, df3$V1, spar=0.3)
plot(x,df3$V1, type = "l", col = rgb(0, 0, 1, 0.3), xlab = "Iteration", ylab = "Cross-Entropy-Loss", ylim = c(0,1))
lines(smoothingSpline, col = "red", lwd = 3)
legend(c(500000, 800000), c(0.88, 1.0), legend=c("originale Daten", "geglättete Daten"),
       col=c("blue", "red"), lty=1:1, cex=1.0)

