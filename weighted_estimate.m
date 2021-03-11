function X_est = weighted_estimate(Xk, weightsk)
%X(k-1) PF tank 1/2/3      after PF_resampling, 1x100
%weights(k-1)              after PF_resampling, 1x100

%Separate the the continuous state
values = Xk; 

%multiply the X(k-1) of each tank by the weights(k-1) to obtain a weighted
%estimate
X_est=values*weightsk';
