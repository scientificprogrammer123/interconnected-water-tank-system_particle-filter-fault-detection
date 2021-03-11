clc;         %clear command window
clearvars;   %clear variables from memory
clf;         %clear current figure window
cla;         %clear axes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% read in data 
M1 = csvread('17_3tank_dynamics_with_leakage2_trial2\pump_PWM_duty_tank3.csv',2,0);
M2 = csvread('17_3tank_dynamics_with_leakage2_trial2\pump_PWM_duty_tank1.csv',2,0);
M3 = csvread('17_3tank_dynamics_with_leakage2_trial2\pressure_sensor_data_tank3.csv',2,0);
M4 = csvread('17_3tank_dynamics_with_leakage2_trial2\pressure_sensor_data_tank2.csv',2,0);
M5 = csvread('17_3tank_dynamics_with_leakage2_trial2\pressure_sensor_data_tank1.csv',2,0);
M6 = csvread('17_3tank_dynamics_with_leakage2_trial2\pulses_counted_out_tank3.csv',2,0);
M7 = csvread('17_3tank_dynamics_with_leakage2_trial2\pulses_counted_out_tank2.csv',2,0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define time start, need to make this more elegent
time_start = 64259;
zero_time_start = 0;
experiment_duration = 217;
time_stop = zero_time_start + experiment_duration;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot tank3's PWM duty cycle vs time
pwm3_hr=M1(:,1);
pwm3_min=M1(:,2);
pwm3_sec=M1(:,3);
pwm3_msec=M1(:,4);
pwm3_level=M1(:,5);
format longg %define data format
[m,n]=size(pwm3_hr);%get data size
pwm3_time=zeros(m,n);
for j=1:m
    pwm3_time(j) = pwm3_msec(j)+pwm3_sec(j)+pwm3_min(j)*60+pwm3_hr(j)*60*60; %calculate time in seconds
end
%% pwm level line 1 is repeated on line 2, so start at row 2, need to fix labview code
pwm3_levels = pwm3_level(2:length(pwm3_level));
%% normalize the PWM levels from 0.43-1.00 to 0.00-1.00
normalized_min = 0.43;
normalized_max = 1.00;
pwm3_levels_normalized = (pwm3_levels-0.43)*(1/0.57);
%% create array to use in estimation
X2=pwm3_levels;
x2=X2(1:length(X2)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot tank1's PWM duty cycles vs time
pwm1_hr=M2(:,1);
pwm1_min=M2(:,2);
pwm1_sec=M2(:,3);
pwm1_msec=M2(:,4);
pwm1_level=M2(:,5);
format longg %define data format
[m,n]=size(pwm1_hr);%get data size
pwm1_time=zeros(m,n);
for j=1:m
    pwm1_time(j) = pwm1_msec(j)+pwm1_sec(j)+pwm1_min(j)*60+pwm1_hr(j)*60*60; %calculate time in seconds
end
%% pwm level line 1 is repeated on line 2, so start at index 2, need to fix labview code
pwm1_levels = pwm1_level(2:length(pwm1_level));
%% normalize the PWM levels
normalized_min = 0.43;
normalized_max = 1.00;
pwm1_levels_normalized = (pwm1_levels-0.43)*(1/0.57);
%% create array to use in estimation
X3=pwm1_levels;
x3=X3(1:length(X3));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot tank3's pressure sensor voltage vs time
pressure_hr_t3=M3(:,6);
pressure_min_t3=M3(:,7);
pressure_sec_t3=M3(:,8);
pressure_msec_t3=M3(:,9);
pressure_voltage_t3=M3(:,11);
[u,v]=size(pressure_hr_t3);
pressure_time_t3=zeros(u,v);
for j=1:size(pressure_hr_t3)
    pressure_time_t3(j) = pressure_msec_t3(j)+pressure_sec_t3(j)+pressure_min_t3(j)*60+pressure_hr_t3(j)*60*60;
end
%% pressure voltage dataset is longer than PWM level dataset
pressure_voltages_t3 = pressure_voltage_t3(1:(length(pressure_hr_t3)));
%% plot water level vs time
F2=figure(1);
subplot(3,1,1);
plot(pressure_voltages_t3,'-o');
axis([zero_time_start time_stop 0 1]);
grid on;
xlabel ('execution time(s)'), ylabel ('voltage(V)'); 
title  ('tank #3 pressure sensor voltage vs. time');
X4=pressure_voltages_t3;
x4=X4(1:length(X4)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot tank2's pressure sensor voltage vs time
pressure_hr_t2=M4(:,6);
pressure_min_t2=M4(:,7);
pressure_sec_t2=M4(:,8);
pressure_msec_t2=M4(:,9);
pressure_voltage_t2=M4(:,11);
[u,v]=size(pressure_hr_t2);
pressure_time_t2=zeros(u,v);
for j=1:size(pressure_hr_t2)
    pressure_time_t2(j) = pressure_msec_t2(j)+pressure_sec_t2(j)+pressure_min_t2(j)*60+pressure_hr_t2(j)*60*60;
end
%% pressure voltage dataset is longer than PWM level dataset 
pressure_voltages_t2 = pressure_voltage_t2(1:(length(pressure_hr_t2)));
%% plot water level vs time
subplot(3,1,2);
plot(pressure_voltages_t2,'-o');
axis([zero_time_start time_stop 0 1]);
grid on
xlabel ('execution time(s)'), ylabel ('voltage(V)');
title  ('tank #3 pressure sensor voltage vs. time');
X5=pressure_voltages_t2;
x5=X5(1:length(X5)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot tank1's pressure sensor voltage vs time
pressure_hr_t1=M5(:,6);
pressure_min_t1=M5(:,7);
pressure_sec_t1=M5(:,8);
pressure_msec_t1=M5(:,9);
pressure_voltage_t1=M5(:,11);
[u,v]=size(pressure_hr_t1);
pressure_time_t1=zeros(u,v);
for j=1:size(pressure_hr_t1)
    pressure_time_t1(j) = pressure_msec_t1(j)+pressure_sec_t1(j)+pressure_min_t1(j)*60+pressure_hr_t1(j)*60*60;
end
%% pressure voltage dataset is longer than PWM level dataset 
pressure_voltages_t1 = pressure_voltage_t1(1:(length(pressure_hr_t1)));
%% plot water level vs time
subplot(3,1,3);
plot(pressure_voltages_t1,'-o');
axis([zero_time_start time_stop 0 1]);
grid on;
xlabel ('execution time(s)'), ylabel ('voltage(V)');
title  ('tank #1 pressure sensor voltage vs. time');
X6=pressure_voltages_t1;
x6=X6(1:length(X6)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot tank3's pulses counted out vs time
pulses_out_hr_t3=M6(:,1);
pulses_out_min_t3=M6(:,2);
pulses_out_sec_t3=M6(:,3);
pulses_out_msec_t3=M6(:,4);
pulses_out_t3=M6(:,5);
[u,v]=size(pulses_out_hr_t3);
for j=1:size(pulses_out_t3)
    pulses_out_time_t3(j) = pulses_out_msec_t3(j)+pulses_out_sec_t3(j)+pulses_out_min_t3(j)*60+pulses_out_hr_t3(j)*60*60;
end
%% pressure voltage dataset is longer than PWM level dataset 
%% plot water level vs time
figure(5)
subplot(2,1,1);
plot(pulses_out_t3,'-o');
axis([zero_time_start time_stop 0 1]);
grid on
xlabel ('execution time(s)'), ylabel ('pulses out');
title  ('tank #3 pulses out vs. time');
X7=pulses_out_t3;
x7=X7(1:length(X7)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot tank2's pulses counted out vs time
pulses_out_hr_t2=M7(:,1);
pulses_out_min_t2=M7(:,2);
pulses_out_sec_t2=M7(:,3);
pulses_out_msec_t2=M7(:,4);
pulses_out_t2=M7(:,5);
[u,v]=size(pulses_out_hr_t2);
for j=1:size(pulses_out_t2)
    pulses_out_time_t2(j) = pulses_out_msec_t2(j)+pulses_out_sec_t2(j)+pulses_out_min_t2(j)*60+pulses_out_hr_t2(j)*60*60;
end
%% pressure voltage dataset is longer than PWM level dataset 
%% plot water level vs time
subplot(2,1,2);
plot(pulses_out_t2,'-o');
axis([zero_time_start time_stop 0 4000]);
grid on
xlabel ('execution time(s)'), ylabel ('pulses out');
title  ('tank #2 pulses out vs. time');
X8=pulses_out_t2;
x8=X8(1:length(X8)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. remap the PWM duty range from 0.43-1.00 to 0.00-1.00
%% 2. pressure voltage dataset is longer than PWM level dataset by 7 seconds,
%%    pad the additional time from when the pump closes to when the pressure 
%%    sensor stop collecting data with zeroes.
zero_pad_array=zeros(length(pressure_hr_t1)-length(pwm1_levels),1);
pwm1_levels_normalized_extended = [pwm1_levels_normalized.' zero_pad_array.'];
pwm3_levels_normalized_extended = [pwm3_levels_normalized.' zero_pad_array.'];
X2=pwm1_levels_normalized_extended;
X3=pwm3_levels_normalized_extended;
%% plot pwm levels vs time
F1=figure(2);
subplot(2,1,1);  
plot(pwm1_levels_normalized_extended,'-o');
axis([zero_time_start time_stop 0.0 1.00]);
grid on;
xlabel ('execution time(s)'), ylabel ('pump pwm duty'); 
title  ('tank1, normalized pump pwm duty vs. time');
subplot(2,1,2);  
plot(pwm3_levels_normalized_extended,'-o')
axis([zero_time_start time_stop 0.0 1.00])
grid on
xlabel ('execution time(s)'), ylabel ('pump pwm duty'), 
title  ('tank3, normalized pump pwm duty vs. time')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute voltage offsets caused by difference in sensor placement
%% from end of experiment when water leve is same between 3 tanks
tank3_initial_voltage = X4(1);
tank2_initial_voltage = X5(1);
tank1_initial_voltage = X6(1);
tank1_offset = tank3_initial_voltage - tank1_initial_voltage
tank2_offset = tank3_initial_voltage - tank2_initial_voltage
tank3_offset = 0;
%% get the coefficients obtained from system identification experiments
alpha_1 = 0.0247;     % tank1-tank2 torticelli coefficient, from experiment 12
alpha_2 = 0.0248;     % tank3-tank2 torticelli coefficient, from experiment 13
beta_1  = 0.0306;     % pwm-to-pump coefficient, from experiment 15
charlie_1 = 0.027783  % tank3's pressure voltage to leakage rate out coefficient, from experiment 8
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chop off the first 5 seconds of data, water is not being pumped into tanks
%% Apply voltage offsets
u1=X2(6:length(X2));  % tank1 pump duty
u3=X3(6:length(X3));  % tank3 pump duty
x3=X4(6:length(X4)) + tank3_offset; % tank3 pressure voltages
x2=X5(6:length(X5)) + tank2_offset; % tank2 pressure voltages
x1=X6(6:length(X6)) + tank1_offset; % tank1 pressure voltages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot the voltages over time after correcting for the pressure sensor position differences
figure(3);
hold on;
plot(x3);
plot(x2);
plot(x1);
hold off;
xlabel ('execution time(s)'), ylabel ('pressure voltage (V)'); 
title  ('tank3 estimated vs actual pressure voltages in vs. time');
legend ('tank3','tank2','tank1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% validate water tank system model using coefficients
x_est1(1)=x1(1);
x_est2(1)=x2(1);
x_est3(1)=x3(1);

%% these coefficients have been hand tuned by prof Raptis to fit to the pressure voltage data
alpha_1   = 0.92 * alpha_1;       % 0.75
alpha_2   = 0.97 * alpha_2;       % 0.75
beta_1    = 1.09 * beta_1;        % 1.10
beta_2    = 1.08 * beta_1;        % 1.20
charlie_1 = 1.08 * charlie_1;     % 1.10, higher coefficient means faster flow out in model
timeOpenLeak2 = 60 + 2 - 5;       % 2 second delay from prompt on screen to rotate tap, chop off 1st 5 datapoints when pump is off
for i = 1:(length(x2) - 1)
    x_est1(i+1) = x_est1(i) + alpha_1 * sign(x_est2(i) - x_est1(i)) * sqrt(abs(x_est1(i) - x_est2(i))) + beta_1 * u1(i);
    if (i < timeOpenLeak2)
        x_est2(i+1) = x_est2(i) + alpha_1 * sign(x_est1(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est1(i))) + alpha_2 * sign(x_est3(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est3(i)));
    else (i >= timeOpenLeak2)
        x_est2(i+1) = x_est2(i) + alpha_1 * sign(x_est1(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est1(i))) + alpha_2 * sign(x_est3(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est3(i))) - charlie_1 * sqrt(abs(x_est2(i) - tank2_offset));
    end
    x_est3(i+1) = x_est3(i) + alpha_2 * sign(x_est2(i) - x_est3(i)) * sqrt(abs(x_est3(i) - x_est2(i))) + beta_2 * u3(i);
end

%% plot
figure(4);
clf;
subplot(3,1,1);
hold on;
plot(x1);
plot(x_est1);
hold off;
xlabel ('execution time(s)'), ylabel ('pressure voltage (V)');
title  ('tank1 estimated vs actual pressure voltages in vs. time');
legend ('actual','estimated');
%%%%%%%%%%%%%%
subplot(3,1,2);
hold on;
plot(x2);
plot(x_est2);
hold off;
xlabel ('execution time(s)'), ylabel ('pressure voltage (V)'); 
title  ('tank2 estimated vs actual pressure voltages in vs. time');
legend ('actual','estimated');
%%%%%%%%%%%%%
subplot(3,1,3);
hold on;
plot(x3);
plot(x_est3);
hold off;
xlabel ('execution time(s)'), ylabel ('pressure voltage (V)'); 
title  ('tank3 estimated vs actual pressure voltages in vs. time');
legend ('actual','estimated');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              particle filter portion of the code starts here 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%clear all
%clc

%% In original particle filter code there are 4 processes:
%  x = blue = uncertain process, the state=f(previous state, noise)
%  z = red = actual process
%  y = yellow = noisy measurement, the output, the state with 
%  x_est = purple = estimated from particle filter

%% For water tank system the processes are:
%  actual pressure voltage in tank1 tank2 tank3
%  estimated pressure voltage in tank1 tank2 tank3
%  system input are pump PWM duty in tank1 and tank3

%% Generate the measurments from a fictional model
%x(k+1)=a*x(k)+w(k) %w(k) is noise, x is the state variable
%y(k)=x(k)+n(k)     %n(k) is noise, y is the output, depends on the state

%% Replace the fictional model with a real model, need all 3 squations since 
%% they are all connected
%x1_pf(i+1) = x1_pf(i) + alpha_1*sign(x2_pf(i)-x1_pf(i))*sqrt(abs(x1_pf(i)-x2_pf(i))) + beta_1*u1(i) + w_1_pf(i);
%x2_pf(i+1) = x2_pf(i) + alpha_1*sign(x1_pf(i)-x2_pf(i))*sqrt(abs(x2_pf(i)-x1_pf(i))) + alpha_2*sign(x3_pf(i)-x2_pf(i))*sqrt(abs(x2_pf(i)-x3_pf(i))) + w_2_pf(i);
%x3_pf(i+1) = x3_pf(i) + alpha_2*sign(x2_pf(i)-x3_pf(i))*sqrt(abs(x3_pf(i)-x2_pf(i))) + beta_2*u3(i) + w_3_pf(i);

%% The process
%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% change for water tank %%%%%
% make the sigmas smaller for water tank
sigma_w_pf=0.01; %sigma for w noise
sigma_n_pf=4;    %sigma for n noise
                 %sigma w is smaller than sigma_n

%%%% change for water tank %%%%%
N=79;
w_1_pf=sigma_w_pf*randn(N,1);  %w noise values for process x1
w_2_pf=sigma_w_pf*randn(N,1);  %w noise values for process x2
w_3_pf=sigma_w_pf*randn(N,1);  %w noise values for process x3
n_1_pf=sigma_n_pf*randn(N,1);  %n noise values for process x1
n_2_pf=sigma_n_pf*randn(N,1);  %n noise values for process x2
n_3_pf=sigma_n_pf*randn(N,1);  %n noise values for process x3

%%%% change for water tank %%%%
x0_1_pf=x1(1); x0_2_pf=x2(1); x0_3_pf=x3(1);
%create the empty arrays
x_1_pf=zeros(N,1); x_2_pf=zeros(N,1); x_3_pf=zeros(N,1);              x_pf=[x_1_pf, x_2_pf, x_3_pf];
x_1_pf(1)=x0_1_pf; x_2_pf(1)=x0_2_pf; x_3_pf(1)=x0_3_pf;       %set initial value of x to x0 + noise

%%%% Test your model %%%%
N=79;
for i=1:N-1
    %state transition
    x_1_pf(i+1) = x_1_pf(i) + alpha_1*sign(x_2_pf(i)-x_1_pf(i))*sqrt(abs(x_1_pf(i)-x_2_pf(i))) + beta_1*u1(i) + w_1_pf(i);
    x_2_pf(i+1) = x_2_pf(i) + alpha_1*sign(x_1_pf(i)-x_2_pf(i))*sqrt(abs(x_2_pf(i)-x_1_pf(i))) + alpha_2*sign(x_3_pf(i)-x_2_pf(i))*sqrt(abs(x_2_pf(i)-x_3_pf(i))) + w_2_pf(i);
    x_3_pf(i+1) = x_3_pf(i) + alpha_2*sign(x_2_pf(i)-x_3_pf(i))*sqrt(abs(x_3_pf(i)-x_2_pf(i))) + beta_2*u3(i) + w_3_pf(i);
    %output emission   
end
%matlab assigns values not reference, so need to re-assign here again
x_pf=[x_1_pf, x_2_pf, x_3_pf];

figure(12)
subplot(3,1,1)
hold on
plot(x_1_pf)
plot(x1)
hold off
subplot(3,1,2)
hold on
plot(x_2_pf)
plot(x2)
hold off
subplot(3,1,3)
hold on
plot(x_3_pf)
plot(x3)
hold off
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% stuff used in particle filter algorithm
M=100; %number of particles

%% Initialize the weights of the partciles
weights_0=(1/M)*ones(1,M); %sum of all weights = 1, each weight is 0.001
weightsk_1=weights_0;      %set it to 1x1000 array of 0.001

%% Initial conditions of the process
X0_1_pf=x1(1)*ones(1,M);
X0_2_pf=x2(1)*ones(1,M);
X0_3_pf=x3(1)*ones(1,M);
Xk_1_1_pf=X0_1_pf;        %1x1000
Xk_1_2_pf=X0_2_pf;        %1x1000
Xk_1_3_pf=X0_3_pf;        %1x1000
 
%% This are our perception about the noise
sigma_w_hat_pf=0.005;     %noise added to the state x,
sigma_n_hat_pf=0.01;      %noise added to the output y, perceived output noise is higher than state noise

%random noise for each particle, for each step in the process
N=212;
w_hat_1_pf=sigma_w_hat_pf*randn(N,M); %79x1000 array of random numbers
w_hat_2_pf=sigma_w_hat_pf*randn(N,M);
w_hat_3_pf=sigma_w_hat_pf*randn(N,M);
n_hat_1_pf=sigma_n_hat_pf*randn(N,M);
n_hat_2_pf=sigma_n_hat_pf*randn(N,M);
n_hat_3_pf=sigma_n_hat_pf*randn(N,M);

%% particle filter estimate of torticelli process
%% change #process from 1 to 3 interconnected 
%These are two tuning parameters. I am not sure what to pick as a value
Ng=1;
Sig=0.001;        %the lower this is, the more the correct particles will be rewarded with the weights
Sig_inv=inv(Sig); %invert Sig=0.01
Sig_inv_matrix=diag([Sig_inv Sig_inv Sig_inv]);
Const=(1/((2*pi)^(Ng/2)))*sqrt(det(Sig)); %what does this constant do?

%% particle filter estimation starts here
for i=1:N                                                   
    %state update
    Xk_1_pf = Xk_1_1_pf + alpha_1*sign(Xk_1_2_pf-Xk_1_1_pf).*sqrt(abs(Xk_1_1_pf-Xk_1_2_pf)) + beta_1*u1(i) + w_hat_1_pf(i,:);
    if   (i <  timeOpenLeak2)
        Xk_2_pf = Xk_1_2_pf + alpha_1*sign(Xk_1_1_pf-Xk_1_2_pf).*sqrt(abs(Xk_1_2_pf-Xk_1_1_pf)) + alpha_2*sign(Xk_1_3_pf-Xk_1_2_pf).*sqrt(abs(Xk_1_2_pf-Xk_1_3_pf)) + w_hat_2_pf(i,:);
    else (i >= timeOpenLeak2)
        Xk_2_pf = Xk_1_2_pf + alpha_1*sign(Xk_1_1_pf-Xk_1_2_pf).*sqrt(abs(Xk_1_2_pf-Xk_1_1_pf)) + alpha_2*sign(Xk_1_3_pf-Xk_1_2_pf).*sqrt(abs(Xk_1_2_pf-Xk_1_3_pf)) - charlie_1*sqrt(abs(Xk_1_2_pf-tank2_offset)) + w_hat_2_pf(i,:);
    end
    Xk_3_pf = Xk_1_3_pf + alpha_2*sign(Xk_1_2_pf-Xk_1_3_pf).*sqrt(abs(Xk_1_3_pf-Xk_1_2_pf)) + beta_2*u3(i) + w_hat_3_pf(i,:);
    Xk_raptis = [Xk_1_pf; Xk_1_pf; Xk_3_pf]; %per prof Rpatis
    %output update
    Yk_1_pf = Xk_1_pf + n_hat_1_pf(i,:);
    Yk_2_pf = Xk_2_pf + n_hat_2_pf(i,:);
    Yk_3_pf = Xk_3_pf + n_hat_3_pf(i,:);
    Yk_raptis = [Yk_1_pf; Yk_2_pf; Yk_3_pf]; %per prof Raptis    
    %update Xk-1 for next loop iteration
    Xk_1_1_pf = Xk_1_pf;
    Xk_1_2_pf = Xk_2_pf;
    Xk_1_3_pf = Xk_3_pf;
      
    Xk = [Xk_1_1_pf; Xk_1_2_pf; Xk_1_3_pf];
    Xk_1 = [Xk_1_1_pf; Xk_1_2_pf; Xk_1_3_pf]; %per prof Raptis, make a 3x1 vector
           
    for j=1:M
        
        %EstError=y(i)-Yk(1,j);%This is the error in estimation. You compare with a measurment rememebr to include the iteration number.
        if (i<N)
            EstError_1_pf=x1(i+1)-Yk_1_pf(1,j);
            EstError_2_pf=x2(i+1)-Yk_1_pf(1,j);
            EstError_3_pf=x3(i+1)-Yk_2_pf(1,j);
        else %(i=N)       
            %EstError_1_pf=EstError_1_pf;
            %EstError_2_pf=EstError_2_pf;
            %EstError_3_pf=EstError_3_pf;
        end
        
        EstError_array = [EstError_1_pf; EstError_2_pf; EstError_3_pf]; %per prof Raptis, make a 3x1 vector

        %per prof Raptis
        weights_inter_raptis(1,j)=exp(-0.5*EstError_array'*Sig_inv_matrix*EstError_array)*Const; %Those are the unnormalized weights. (1,nParticles)    %compute weight matrix based on the EstError
        %end per prof raptis        
    end 
    
    %to avoid degeneracy
    smallConst=ones(1,M)*1e-9; %This is a very small constant to add to the weight update. I am not sure why is used    
    weightsk=weights_inter_raptis.*weightsk_1+smallConst; %dimensions (1,nParticles), make sure all weights are non-zero
    
    %Normalize the weights
    weightsk=(1/sum(weightsk))*weightsk;   
    %Resampling
    wSum_Sq=weightsk*weightsk';   
        
    %This is the tricky part. Look in the books for details
    Thr=0.65; %resampling threshold        
    [Xk_1,weightsk_1]=PF_resampling(Xk,weightsk,M,Thr); % This is completely taken from Doug, I don't know what is happening
    X_est_1_pf(i)=weighted_estimate(Xk_1_pf,weightsk);
    X_est_2_pf(i)=weighted_estimate(Xk_2_pf,weightsk);
    X_est_3_pf(i)=weighted_estimate(Xk_3_pf,weightsk);  

    %{
    %plot the particles for the process
    figure(13)
    clf;
    subplot(3,1,1)   
    hold on
    plot(x1)
    plot(i*ones(1,M),Xk_1_1_pf,'o')
    hold off
    %%%%
    subplot(3,1,2)   
    hold on
    plot(x2)
    plot(i*ones(1,M),Xk_1_2_pf,'o')
    hold off
    %%%%
    subplot(3,1,3)
    hold on
    plot(x3)
    plot(i*ones(1,M),Xk_1_3_pf,'o')
    hold off
    pause(0.01)

    %plot steps of the estimate against the process
    figure(14)
    clf
    subplot(3,1,1) 
    hold on
    stem(x2(i),1,'r')
    stem(X_est_2_pf(i),1,'g')
    stem(Xk_1_2_pf,weightsk)    
    hold off
    xlim([0 1])
    %%%%
    subplot(3,1,2) 
    hold on
    stem(x2(i),1,'r')
    stem(X_est_2_pf(i),1,'g')
    stem(Xk_1_2_pf,weightsk)
    hold off
    xlim([0 1])
    %%%%
    subplot(3,1,3) 
    hold on
    stem(x3(i),1,'r')
    stem(X_est_3_pf(i),1,'g')
    stem(Xk_1_3_pf,weightsk)
    hold off
    xlim([0 1])
    pause(0.01)
    %}
end

%% plot all 4 figures together
figure(10)
subplot(1,3,1);
plot(1:1:N,x1,'--',1:1:N,x_est1,':',1:1:N,X_est_1_pf,'-.');
axis([zero_time_start time_stop 0 1]);
grid on;
xticks([0 20 40 60 80 100 120 140 160 180 200]);
yticks([0 0.2 0.4 0.6 0.8 1.0]);
legend('measured','modelled','estimated with particle filter')
xlabel ('time(s)'), ylabel ('voltage (V)'); 
title  ('tank1 pressure sensor voltage vs. time');
%figure(11)
subplot(1,3,2);
plot(1:1:N,x2,'--',1:1:N,x_est2,':',1:1:N,X_est_2_pf,'-.')
axis([zero_time_start time_stop 0 1]);
grid on;
xticks([0 20 40 60 80 100 120 140 160 180 200]);
yticks([0 0.2 0.4 0.6 0.8 1.0]);
legend('measured','modelled','estimated with particle filter')
xlabel ('time(s)'), ylabel ('voltage (V)'); 
title  ('tank2 pressure sensor voltage vs. time');
%figure(12)
subplot(1,3,3);
plot(1:1:N,x3,'--',1:1:N,x_est3,':',1:1:N,X_est_3_pf,'-.')
axis([zero_time_start time_stop 0 1]);
grid on;
xticks([0 20 40 60 80 100 120 140 160 180 200]);
yticks([0 0.2 0.4 0.6 0.8 1.0]);
legend('measured','modelled','estimated with particle filter')
xlabel ('time(s)'), ylabel ('voltage (V)'); 
title  ('tank3 pressure sensor voltage vs. time');