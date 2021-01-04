require(PerMallows)

setwd("C:/Users/sqshy/Desktop/University/Fifth Year/research/DistortionSim/rdata")

theta_vals <- c(1,0.2,5)
trial_sizes <- c(5,10,20,50,100)
num_iterations <- 100
file_vector <- c("ord_n5_theta1.txt", "ord_n10_theta1.txt", "ord_n20_theta1.txt", "ord_n50_theta1.txt", "ord_n100_theta1.txt",
                 "ord_n5_theta0.2.txt", "ord_n10_theta0.2.txt", "ord_n20_theta0.2.txt", "ord_n50_theta0.2.txt", "ord_n100_theta0.2.txt",
                 "ord_n5_theta5.txt", "ord_n10_theta5.txt", "ord_n20_theta5.txt", "ord_n50_theta5.txt", "ord_n100_theta5.txt")
file_names <- matrix(file_vector, nrow = length(theta_vals), ncol = length(trial_sizes), byrow=TRUE)

set.seed(42)

for(k in 1:length(theta_vals)){
	for(i in 1:length(trial_sizes)){
            
		A = rmm(n=trial_sizes[i], sigma0=1:trial_sizes[i],theta = theta_vals[k])
		write.table(A,file=file_names[k,i], append=FALSE, sep = ",", row.names=FALSE, col.names = FALSE)

			for(j in 1:(num_iterations-1)){

				A = rmm(n=trial_sizes[i], sigma0=1:trial_sizes[i],theta = theta_vals[k])
		            write.table(A,file=file_names[k,i], append=TRUE, sep = ",", row.names=FALSE, col.names = FALSE)
		}
	}
}