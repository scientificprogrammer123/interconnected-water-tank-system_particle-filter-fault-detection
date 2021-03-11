%% clear window, varlables, figures, axes
clc;         %clear command window
clearvars;   %clear variables from memory
clf;         %clear current figure window
cla;         %clear axes

%% read in data 
%{
% for windows
M1 = csvread('17_3tank_dynamics_with_leakage2_trial2\pump_PWM_duty_tank3.csv',2,0);
M2 = csvread('17_3tank_dynamics_with_leakage2_trial2\pump_PWM_duty_tank1.csv',2,0);
M3 = csvread('17_3tank_dynamics_with_leakage2_trial2\pressure_sensor_data_tank3.csv',2,0);
M4 = csvread('17_3tank_dynamics_with_leakage2_trial2\pressure_sensor_data_tank2.csv',2,0);
M5 = csvread('17_3tank_dynamics_with_leakage2_trial2\pressure_sensor_data_tank1.csv',2,0);
M6 = csvread('17_3tank_dynamics_with_leakage2_trial2\pulses_counted_out_tank3.csv',2,0);
M7 = csvread('17_3tank_dynamics_with_leakage2_trial2\pulses_counted_out_tank2.csv',2,0);
%}
% for linux
M1 = csvread('17_3tank_dynamics_with_leakage2_trial2/pump_PWM_duty_tank3.csv',2,0);
M2 = csvread('17_3tank_dynamics_with_leakage2_trial2/pump_PWM_duty_tank1.csv',2,0);
M3 = csvread('17_3tank_dynamics_with_leakage2_trial2/pressure_sensor_data_tank3.csv',2,0);
M4 = csvread('17_3tank_dynamics_with_leakage2_trial2/pressure_sensor_data_tank2.csv',2,0);
M5 = csvread('17_3tank_dynamics_with_leakage2_trial2/pressure_sensor_data_tank1.csv',2,0);
M6 = csvread('17_3tank_dynamics_with_leakage2_trial2/pulses_counted_out_tank3.csv',2,0);
M7 = csvread('17_3tank_dynamics_with_leakage2_trial2/pulses_counted_out_tank2.csv',2,0);

%% define time start
time_start          = 64259;
zero_time_start     = 0;
experiment_duration = 217;
time_stop           = zero_time_start + experiment_duration;

%% extract and plot PWM duty cycle vs time for tank3
format longg;
pwm3_hr    = M1(:,1);
pwm3_min   = M1(:,2);
pwm3_sec   = M1(:,3);
pwm3_msec  = M1(:,4);
pwm3_level = M1(:,5);
[m,n]      = size(pwm3_hr); %get data size
pwm3_time  = zeros(m,n);
for j = 1:m
    pwm3_time(j) = pwm3_msec(j) + pwm3_sec(j) + pwm3_min(j)*60 + pwm3_hr(j)*60*60; %calculate time in seconds
end
%pwm level line 1 is repeated on line 2, so start at row 2, fix it in labview
pwm3_levels    = pwm3_level(2:length(pwm3_level));
%normalize the PWM levels from 0.43-1.00 to 0.00-1.00
normalized_min = 0.43;
normalized_max = 1.00;
pwm3_levels_normalized = (pwm3_levels-0.43) * (1/0.57);
%create array to use in estimation
X2 = pwm3_levels;
x2 = X2(1:length(X2)); 

%% extract and plot PWM duty cycles vs time for tank1
format longg; 
pwm1_hr    = M2(:,1);
pwm1_min   = M2(:,2);
pwm1_sec   = M2(:,3);
pwm1_msec  = M2(:,4);
pwm1_level = M2(:,5);
[m,n]      = size(pwm1_hr);%get data size
pwm1_time  = zeros(m,n);
for j = 1:m
    pwm1_time(j) = pwm1_msec(j) + pwm1_sec(j) + pwm1_min(j)*60 + pwm1_hr(j)*60*60; %calculate time in seconds
end
%pwm level line 1 is repeated on line 2, so start at index 2, need to fix labview code
pwm1_levels    = pwm1_level(2:length(pwm1_level));
%normalize the PWM levels from 0.43-1.00 to 0.00-1.00
normalized_min = 0.43;
normalized_max = 1.00;
pwm1_levels_normalized = (pwm1_levels-0.43) * (1/0.57);
%create array to use in estimation
X3 = pwm1_levels;
x3 = X3(1:length(X3));

%% extract and plot pressure sensor voltage vs time for tank3
pressure_hr_t3      = M3(:,6);
pressure_min_t3     = M3(:,7);
pressure_sec_t3     = M3(:,8);
pressure_msec_t3    = M3(:,9);
pressure_voltage_t3 = M3(:,11);
[u,v] = size(pressure_hr_t3);
pressure_time_t3    = zeros(u,v);
for j = 1:size(pressure_hr_t3)
    pressure_time_t3(j) = pressure_msec_t3(j) + pressure_sec_t3(j) + pressure_min_t3(j)*60 + pressure_hr_t3(j)*60*60;
end
%pressure voltage dataset is longer than PWM level dataset
pressure_voltages_t3 = pressure_voltage_t3(1:(length(pressure_hr_t3)));
%plot water level vs time
%{
F2=figure(1);
subplot(3,1,1);
plot(pressure_voltages_t3,'-o');
axis([zero_time_start time_stop 0 1]);
grid on;
xlabel ('execution time(s)'), ylabel ('voltage(V)'); 
title  ('tank #3 pressure sensor voltage vs. time');
%}
X4 = pressure_voltages_t3;
x4 = X4(1:length(X4)); 

%% extract and plot pressure sensor voltage vs time for tank2
pressure_hr_t2      = M4(:,6);
pressure_min_t2     = M4(:,7);
pressure_sec_t2     = M4(:,8);
pressure_msec_t2    = M4(:,9);
pressure_voltage_t2 = M4(:,11);
[u,v] = size(pressure_hr_t2);
pressure_time_t2    = zeros(u,v);
for j=1:size(pressure_hr_t2)
    pressure_time_t2(j) = pressure_msec_t2(j) + pressure_sec_t2(j) + pressure_min_t2(j)*60 + pressure_hr_t2(j)*60*60;
end
%pressure voltage dataset is longer than PWM level dataset 
pressure_voltages_t2 = pressure_voltage_t2(1:(length(pressure_hr_t2)));
%plot water level vs time
%{
subplot(3,1,2);
plot(pressure_voltages_t2,'-o');
axis([zero_time_start time_stop 0 1]);
grid on
xlabel ('execution time(s)'), ylabel ('voltage(V)');
title  ('tank #3 pressure sensor voltage vs. time');
%}
X5 = pressure_voltages_t2;
x5 = X5(1:length(X5)); 

%% extract and plot pressure sensor voltage vs time for tank1
pressure_hr_t1      = M5(:,6);
pressure_min_t1     = M5(:,7);
pressure_sec_t1     = M5(:,8);
pressure_msec_t1    = M5(:,9);
pressure_voltage_t1 = M5(:,11);
[u,v]=size(pressure_hr_t1);
pressure_time_t1    = zeros(u,v);
for j=1:size(pressure_hr_t1)
    pressure_time_t1(j) = pressure_msec_t1(j) + pressure_sec_t1(j) + pressure_min_t1(j)*60 + pressure_hr_t1(j)*60*60;
end
%pressure voltage dataset is longer than PWM level dataset, chop off the excess
pressure_voltages_t1 = pressure_voltage_t1(1:(length(pressure_hr_t1)));
%plot water level vs time
%{
subplot(3,1,3);
plot(pressure_voltages_t1,'-o');
axis([zero_time_start time_stop 0 1]);
grid on;
xlabel ('execution time(s)'), ylabel ('voltage(V)');
title  ('tank #1 pressure sensor voltage vs. time');
%}
X6 = pressure_voltages_t1;
x6 = X6(1:length(X6)); 

%% extract and plot pulses counted out vs time for tank3
%{
pulses_out_hr_t3=M6(:,1);
pulses_out_min_t3=M6(:,2);
pulses_out_sec_t3=M6(:,3);
pulses_out_msec_t3=M6(:,4);
pulses_out_t3=M6(:,5);
[u,v]=size(pulses_out_hr_t3);
for j=1:size(pulses_out_t3)
    pulses_out_time_t3(j) = pulses_out_msec_t3(j)+pulses_out_sec_t3(j)+pulses_out_min_t3(j)*60+pulses_out_hr_t3(j)*60*60;
end
%pressure voltage dataset is longer than PWM level dataset 
pressure_voltages_t3 = pressure_voltage_t3(1:(length(pressure_hr_t3)));
%plot water level vs 
%%{
figure(5)
subplot(2,1,1);
plot(pulses_out_t3,'-o');
axis([zero_time_start time_stop 0 1]);
grid on
xlabel ('execution time(s)'), ylabel ('pulses out');
title  ('tank #3 pulses out vs. time');
%%}
X7=pulses_out_t3;
x7=X7(1:length(X7));
%}

%% extract and plot pulses counted out vs time for tank2
%{
pulses_out_hr_t2=M7(:,1);
pulses_out_min_t2=M7(:,2);
pulses_out_sec_t2=M7(:,3);
pulses_out_msec_t2=M7(:,4);
pulses_out_t2=M7(:,5);
[u,v]=size(pulses_out_hr_t2);
for j=1:size(pulses_out_t2)
    pulses_out_time_t2(j) = pulses_out_msec_t2(j)+pulses_out_sec_t2(j)+pulses_out_min_t2(j)*60+pulses_out_hr_t2(j)*60*60;
end
%pressure voltage dataset is longer than PWM level dataset 
%plot water level vs time
subplot(2,1,2);
plot(pulses_out_t2,'-o');
axis([zero_time_start time_stop 0 4000]);
grid on
xlabel ('execution time(s)'), ylabel ('pulses out');
title  ('tank #2 pulses out vs. time');
X8=pulses_out_t2;
x8=X8(1:length(X8)); 
%}

%% pressure voltage dataset is longer than PWM level dataset by 7 seconds,
%% pad the additional time from when the pump closes to when the pressure 
%% sensor stop collecting data with zeroes.
zero_pad_array = zeros(length(pressure_hr_t1) - length(pwm1_levels),1);
pwm1_levels_normalized_extended = [pwm1_levels_normalized.' zero_pad_array.'];
pwm3_levels_normalized_extended = [pwm3_levels_normalized.' zero_pad_array.'];
X2 = pwm1_levels_normalized_extended;
X3 = pwm3_levels_normalized_extended;
%plot pwm levels vs time after data padding
%{
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
%}

%% Compute the voltage offsets caused by difference in sensor height placement
%% from the end of experiment when water level is same between 3 tanks
tank3_initial_voltage = X4(1);
tank2_initial_voltage = X5(1);
tank1_initial_voltage = X6(1);
tank1_offset = tank3_initial_voltage - tank1_initial_voltage
tank2_offset = tank3_initial_voltage - tank2_initial_voltage
tank3_offset = 0;

%% Chop off the first 5 seconds of data, during this time water is not being 
%% pumped into tanks, 
%% apply voltage offsets
u1 = X2(6:length(X2));                % tank1 pump duty
u3 = X3(6:length(X3));                % tank3 pump duty
x3 = X4(6:length(X4)) + tank3_offset; % tank3 pressure voltages
x2 = X5(6:length(X5)) + tank2_offset; % tank2 pressure voltages
x1 = X6(6:length(X6)) + tank1_offset; % tank1 pressure voltages

%% Plot the voltages over time after correcting for the pressure sensor 
%% position differences
%{
figure(3);
hold on;
plot(x3);
plot(x2);
plot(x1);
hold off;
xlabel ('execution time(s)'), ylabel ('pressure voltage (V)'); 
title  ('tank3 estimated vs actual pressure voltages in vs. time');
legend ('tank3','tank2','tank1');
%}

%% Initial values of the estimated tank voltages, these will be used to build 
%% a model of the voltages in the tank in time.
x_est1(1) = x1(1);
x_est2(1) = x2(1);
x_est3(1) = x3(1);

%% These are the coefficients obtained from system identification experiments.
alpha_1   = 0.0247;               % tank1-tank2 torticelli coefficient, from experiment 12
alpha_2   = 0.0248;               % tank3-tank2 torticelli coefficient, from experiment 13
beta_1    = 0.0306;               % pwm-to-pump coefficient, from experiment 15
charlie_1 = 0.027783;             % tank3's pressure voltage to leakage rate out coefficient, from experiment 8
%These coefficients have been hand tuned by prof Raptis to fit to the 
%pressure voltage data. They are used in the model of the tank voltages.
alpha_1   = 0.92 * alpha_1       % 0.75
alpha_2   = 0.97 * alpha_2       % 0.75
beta_1    = 1.09 * beta_1        % 1.10
beta_2    = 1.08 * beta_1        % 1.20
charlie_1 = 1.08 * charlie_1     % 1.10, higher coefficient means faster flow out in model
timeOpenLeak2 = 60 + 2 - 5;       % 2 second delay from prompt on screen to rotate tap, chop off 1st 5 datapoints when pump is off

%% First create the model of the system using torricelli equation and the 
%% coefficients from system identification
for i = 1:(length(x2) - 1)
    
    %x_est1 is the estimated state of the system for the first tank
    x_est1(i+1) = x_est1(i) + alpha_1 * sign(x_est2(i) - x_est1(i)) * sqrt(abs(x_est1(i) - x_est2(i))) + beta_1 * u1(i);
    
    %x_est2 is the estimated state of the system for the second tank
    if (i < timeOpenLeak2);  %if the time of system is less than leakage occuring, then the model does not care about the leakage parts 
        x_est2(i+1) = x_est2(i) + alpha_1 * sign(x_est1(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est1(i))) + alpha_2 * sign(x_est3(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est3(i)));
        %what is Qleak used for?
        Qleak(i) = 0; 
    else (i >= timeOpenLeak2); %this is a statement, not a comparison
        %if the time of system is greater than time of leakage occuring, then the model has a leakage term, which is used to model the leakage term, this is charlie_1
        x_est2(i+1) = x_est2(i) + alpha_1 * sign(x_est1(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est1(i))) + alpha_2 * sign(x_est3(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est3(i))) - charlie_1 * sqrt(abs(x_est2(i) - tank2_offset));       
        %this Qleak(i) term is implemented here, what is it for? it is not
        %used in the model.
        Qleak(i)=charlie_1 * sqrt(abs(x_est2(i) - tank2_offset));
    end   
    %what are Q1 and Q2 terms here? they are not used in the estimates
    %Q1 term is related to tank 1?
    Q1(i) = alpha_1 * sign(x_est1(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est1(i)));
    %Q2 term is related to tank 2?
    Q2(i) = alpha_2 * sign(x_est3(i) - x_est2(i)) * sqrt(abs(x_est2(i) - x_est3(i)));
    
    %x_est3 is the estimated state of the system for the third tank,
    %it does not have a leakage term associated with it
    x_est3(i+1) = x_est3(i) + alpha_2 * sign(x_est2(i) - x_est3(i)) * sqrt(abs(x_est3(i) - x_est2(i))) + beta_2 * u3(i);

end

%% Plot the measured voltages and modelled voltages
figure(4);
%figure('units','normalized','outerposition',[0 0 1 1]) %make the plot full screen
clf;
subplot(3,1,1);
hold on;
plot(x1, '--');
plot(x_est1, '-.');
hold off;
grid on; grid minor;
axis([zero_time_start time_stop 0 1]);
xlabel ('time(s)'), ylabel ('voltage (V)');
title  ('tank 1 voltage vs. time');
legend ('measured voltage','modelled voltage');
legend('location', 'south');
%%%%%%%%%%%%%%
subplot(3,1,2);
hold on;
plot(x2, '--');
plot(x_est2, '-.');
hold off;
grid on; grid minor;
axis([zero_time_start time_stop 0 1]);
xlabel ('time(s)'), ylabel ('voltage (V)'); 
title  ('tank 2 voltage vs. time');
legend ('measured voltage','modelled voltage');
legend('location', 'south');
%%%%%%%%%%%%%
subplot(3,1,3);
hold on;
plot(x3, '--');
plot(x_est3, '-.');
hold off;
grid on; grid minor;
axis([zero_time_start time_stop 0 1]);
xlabel ('time(s)'), ylabel ('voltage (V)'); 
title  ('tank 3 voltage vs. time');
legend ('measured voltage','modelled voltage');
legend('location', 'south');
style = hgexport('factorystyle');
style.Color = 'gray';
colormap(gray);
%hgexport(gcf,'figure_3_1.eps',style);
%saveas(gcf,'figure_3_1.jpg');
%cmap = colormap('gray');
%imwrite(gcf,cmap, 'figure_3_1.jpg', 'jpeg');
%I = ind2gray(grayimage,cmap);
%imwrite(I,'imagename.jpg');
saveas(gcf,'figure_3_1.jpg');
%cmap = colormap('gray');
%imwrite(gcf,cmap, 'figure_3_1.jpg', 'jpeg');
%I = ind2gray(grayimage,cmap);
%imwrite(I,'imagename.jpg');
contents = dir('figure_3_1.jpg') % or whatever the filename extension is
for k = 1:numel(contents)
  filename   = contents(k).name;
  rgbImg     = imread(filename);
  gsImg      = rgb2gray(rgbImg);
  [~,name,~] = fileparts(filename);
  gsFilename = sprintf('%s_gs.jpg', name);
  imwrite(gsImg,gsFilename);
end

%% Original particle filter code
%{
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
% x(k+1)=a*x(k)+w(k) %w(k) is noise, x is the state variable
% y(k)=x(k)+n(k)     %n(k) is noise, y is the output, depends on the state

%% Replace the fictional model with a real model, need all 3 squations since 
%% they are all connected
% x1_pf(i+1) = x1_pf(i) + alpha_1*sign(x2_pf(i)-x1_pf(i))*sqrt(abs(x1_pf(i)-x2_pf(i))) + beta_1*u1(i) + w_1_pf(i);
% x2_pf(i+1) = x2_pf(i) + alpha_1*sign(x1_pf(i)-x2_pf(i))*sqrt(abs(x2_pf(i)-x1_pf(i))) + alpha_2*sign(x3_pf(i)-x2_pf(i))*sqrt(abs(x2_pf(i)-x3_pf(i))) + w_2_pf(i);
% x3_pf(i+1) = x3_pf(i) + alpha_2*sign(x2_pf(i)-x3_pf(i))*sqrt(abs(x3_pf(i)-x2_pf(i))) + beta_2*u3(i) + w_3_pf(i);

%% The process
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

%% Number of steps in the process
N = 212; %number of steps in the process

%% Specify the number of particles and the weights
M = 100;                       %number of particles = 10, you get problems
weights_0 = (1/M) * ones(1,M); %sum of all weights = 1, each weight is 0.1
weightsk_1 = weights_0;        %initalize weights(k-1) to 1x10 array of 0.1

%% Initial conditions of the process
X0_1_pf = x1(1)*ones(1,M);             %initial value of the particles for tank1
X0_2_pf = x2(1)*ones(1,M);             %initial value of the particles for tank2
X0_3_pf = x3(1)*ones(1,M);             %initial value of the particles for tank3
xk_1_1_pf = X0_1_pf;                   %Xk-1, i.e previous value of the particles, for tank1
xk_1_2_pf = X0_2_pf;                   %Xk-1, i.e previous value of the particles, for tank2
xk_1_3_pf = X0_3_pf;                   %Xk-1, i.e previous value of the particles, for tank3
X_boolean_2_0 = zeros(2,M);
X_boolean_2_0(1,:) = ones(1,M);        %boolean states for tank2,   10 particles, all initialized to (1,0)
X_boolean_1_t2_pf = X_boolean_2_0(:,1); %boolean states-1 for tank2, 10 particles, all initialized to (0,1)
 
%% This are our perception about the noise
sigma_w_hat_pf = 0.01;       %noise added to the state x,
sigma_n_hat_pf = 0.001;      %noise added to the output y,

%% A random noise for each particle, for each step in the process
w_hat_1_pf = sigma_w_hat_pf * randn(N,M); %212x10 array of random numbers
w_hat_2_pf = sigma_w_hat_pf * randn(N,M); 
w_hat_3_pf = sigma_w_hat_pf * randn(N,M);
n_hat_1_pf = sigma_n_hat_pf * randn(N,M);
n_hat_2_pf = sigma_n_hat_pf * randn(N,M);
n_hat_3_pf = sigma_n_hat_pf * randn(N,M);

%% Particle filter estimate of torticelli process
%% these are two tuning parameters
Ng = 1;                                           %used in building the const
Sig = 0.001;                                      %the lower this is, the more the correct particles will be rewarded with the weights
Sig_inv = inv(Sig);                               %=1000
Sig_inv_matrix = diag([Sig_inv Sig_inv Sig_inv]); %diagonal 3x3 matrix for calculation, 1000 for diagonal elements
Const = (1/((2*pi)^(Ng/2))) * sqrt(det(Sig));     %what does this constant do? is it used in the modelling of the software? no, then why is is here?

%% particle filter estimation starts here
for i = 1:N-1  

    %{
    %if i==9
    if i==210
        break;
    end
    %prior belief
    weightsk_1;
    %}
    
    %state update
    xk_1_pf = xk_1_1_pf + alpha_1 * sign(xk_1_2_pf - xk_1_1_pf).* sqrt(abs(xk_1_1_pf - xk_1_2_pf)) + beta_1 * u1(i) + w_hat_1_pf(i,:);    
    xk_2_pf = xk_1_2_pf + alpha_1 * sign(xk_1_1_pf - xk_1_2_pf).* sqrt(abs(xk_1_2_pf - xk_1_1_pf)) + alpha_2 * sign(xk_1_3_pf - xk_1_2_pf).* sqrt(abs(xk_1_2_pf - xk_1_3_pf)) - X_boolean_1_t2_pf(2,:).* (charlie_1 * sqrt(abs(xk_1_2_pf - tank2_offset))) + w_hat_2_pf(i,:);   
    %xk_t2pf has a term associated with it, it is X_boolean_1_2_pf(2,:).*( charlie_1*sqrt(abs(xkm1_t2pf-tank2_offset)))
    %X_boolean_1_2_pf(2,:) is a 1x100 array of 0/1, where 0 means healthy and 1 means faulty. 
    xk_3_pf = xk_1_3_pf + alpha_2 * sign(xk_1_2_pf - xk_1_3_pf).* sqrt(abs(xk_1_3_pf - xk_1_2_pf)) + beta_2 * u3(i) + w_hat_3_pf(i,:);
 
    %boolean state update
    %the mean is set at -0.65, and then the random number is added to it
    %the end result is a random noise set spread between -1.3 and 0
    noiseBoolean = -0.65 * ones(2,M) + 1.3 * rand(2,M); %why is this important?
    %noiseBoolean = -0.60 * ones(2,M) + 1.2 * rand(2,M); %why is this important?
    %noiseBoolean = -0.55 * ones(2,M) + 1.1 * rand(2,M); %why is this important?
    %noiseBoolean = -0.70 * ones(2,M) + 1.4 * rand(2,M); %why is this important?
    %noiseBoolean = -0.75 * ones(2,M) + 1.5 * rand(2,M); %why is this important?

    %noiseBoolean = -0.95 * ones(2,M) + 1.8 * rand(2,M); %why is this important?
    %noiseBoolean = -0.5 * ones(2,M) + 1.0 * rand(2,M);
    %noiseBoolean = 0 * ones(2,M) + 0 * rand(2,M);
    
    %check boolean state, if boolean state is closer to healthy then set it
    %to [1,0], else set it to [0,1]
    for j = 1:M   
        X_boolean_t2_perturbed = X_boolean_1_t2_pf + noiseBoolean; %add boolean to noiseboolean
        %X_boolean_t2_perturbed = X_boolean_1_2_pf;
        %is X_boolean_t2_perturbed closer to [1 0] or [0 1]?
        if norm(X_boolean_t2_perturbed(:,j) - [1 0]') <= norm(X_boolean_t2_perturbed(:,j) - [0 1]') %check if it is closer to 10-healthy or 01-unhealthy
            X_boolean_t2_pf(:,j) = [1 0]'; %healthy
        else
            X_boolean_t2_pf(:,j) = [0 1]'; %unhealthy
        end
    end
   
    %output update
    %output-1 equals to state-1 + noise
    Yk_1_pf   = xk_1_pf + n_hat_1_pf(i,:);
    Yk_2_pf   = xk_2_pf + n_hat_2_pf(i,:);
    Yk_3_pf   = xk_3_pf + n_hat_3_pf(i,:);
    Yk_raptis = [Yk_1_pf; Yk_2_pf; Yk_3_pf]; %per prof Raptis, combine into array   
     
    %compute errors for each particle, use it to construct the current
    %measurement of the particle weights.
    %M is the number of particles = 10
    for j = 1:M 
        
        %error between measured and modelled
        %x1/2/3(i)   = measurement
        %Yk_1/2/3_pf = output of model        
        EstError_1_pf = x1(i) - Yk_1_pf(1,j); 
        EstError_2_pf = x2(i) - Yk_2_pf(1,j);
        EstError_3_pf = x3(i) - Yk_3_pf(1,j);
        
        %compute absolute error of the above
        %[abs(estimated error of tank1), abs(estimated_error of tank3), abs(estimated error of tank3)]
        %Error1(j,i) = abs(EstError_t1pf);   
        %Error2(j,i) = abs(EstError_t2pf);
        %Error3(j,i) = abs(EstError_t3pf);
        EstError_array = [EstError_1_pf; EstError_2_pf; EstError_3_pf]; %per prof Raptis, make a 3x1 vector
       
        %This is the likelihood function. 
        %
        %It assigns the weights to the particle based on the error between 
        %the particle and the measurement.
        %
        %It is some sort of a gaussian function, the larger the error, the
        %smaller the weight.
        %
        %Sig_inv_matrix = identity matrix * sig_inv(1000)
        %
        %exp(-0.5*error'*Sig_inv_matrix*error) means the larger the
        %sig_inv_matrix or the error, the smaller the weights_inter value is.
        %This is to be expected, since a particle with larger error should
        %have a smaller weight.
        weights_inter(1,j) = exp(-0.5 * EstError_array' * Sig_inv_matrix * EstError_array) * Const;            
        
    end
    
    sum(weights_inter) %0.03-0.09, not 1
        
    %To avoid degeneracy, add 1e-9 to each particle.
    %Degeneracy is usually explained as the all the weights becoming extremely 
    %small and it becomes very sensitive due to the peaked likelihood,
    %explained here:
    %https://stats.stackexchange.com/questions/270673/particle-degeneracy-variance-of-the-weights
    smallConst = ones(1,M) * 1e-9; 
    
    %This is the posterior belief.
    %It is constructed from the prior belief(Xk-1) and the current
    %measurement (weights_inter{mediate}).
    %weightsk = weights_inter.* weightsk_1 + smallConst; 
    weightsk = weights_inter.* weightsk_1;
    
    sum(weightsk); %goes from 0.007 to 0.07, because it is the weights(k-1)
                  %multiplied by weights_intermediate as obtained from the 
                  %likelihood function above
    
    %normalize the weights(k) so the sum equals to 1
    weightsk = (1/sum(weightsk)) * weightsk;      
    
    %construct Xk for PF_resampling function
    %Xk is an array of the Xk_?_pf for the three tanks, plus 
    Xk = [xk_1_pf; xk_2_pf; xk_3_pf; X_boolean_t2_pf];
    
    %resampling threshold
    Thr = 0.65;
    
    %resampling function.
    %inputs are:
    %Xk = [xk_t1pf; xk_t2pf; xk_t3pf; X_boolean_t2_pf]
    %weightsk = 
    %M = 10
    %Thr = 0.65
    %PFs are suppose to be survival of the fittest, this is implemented in
    %the PF_resampling function.
    %The particles with higher importance than uniform distribution are
    %picked up and inserted into X(k-1) to be used in the next estimation
    %step
    %i_main = i;
    %i_main;
    %fileID = fopen('exp.txt','a');
    %fprintf(fileID, 'index in main function is %d\n', i_main)
    %fclose(fileID);
    [Xk_1, weightsk_1] = PF_resampling(Xk, weightsk, M, Thr, i); 
       
    %extract X(k-1) for each tank from output of PF_resampling
    xk_1_1_pf = Xk_1(1, :);
    xk_1_2_pf = Xk_1(2, :);
    xk_1_3_pf = Xk_1(3, :);
    
    %update X_boolean(k-1) for tank 2 from output of PF_resampling,   
    X_boolean_1_t2_pf = Xk_1(4:5, :);
    
    %assign weights(k-1) to weights(k) for next iteration 
    weights(:, i) = weightsk_1';
    
    %compute the weighted estimates
    X_est_1_pf(i) = weighted_estimate(xk_1_1_pf, weightsk_1);
    X_est_2_pf(i) = weighted_estimate(xk_1_2_pf, weightsk_1);
    X_est_3_pf(i) = weighted_estimate(xk_1_3_pf, weightsk_1);  
    
    %this is the failure probability for the ith iteration
    %need to do this for every iteration
    %X_boolean_1_2_pf(2,:)
    %weightsk_1' transpose
    %sum(weightsk_1)
    E_failure_2_abrupt_fault(i) = X_boolean_1_t2_pf(2,:) * weightsk_1' / sum(weightsk_1);
    
    %{
    %plot the particle distributions before resampling
    figure(7);
    %hold on;
    plot(xk_2_pf, weightsk,'o');
    xlim([0 1]);
    ylim([0 0.05]);    
    xlabel('particle voltage (v)','FontSize',12);
    ylabel('particle weight','FontSize',12);
    title('Particle weight vs. particle voltage, tank 2, experiment #1, pre-resampling');
    grid on; grid minor;
    %legendInfo = ['time index', num2str(i)];
    %legend(legendInfo)
    %legend('location', 'northeast');
    %saveas(gcf,'particle_PDF_%d','epsc',i)
    %print(figure(7), '-depsc', sprintf('particle_weights%d.eps', i));
    style = hgexport('factorystyle');
    style.Color = 'gray';
    if i==55
      hgexport(gcf,'ex1_particle_distribution55_pre_resampling.eps',style);
    end
    if i==56
      hgexport(gcf,'ex1_particle_distribution56_pre_resampling.eps',style);
    end
    if i==57
      hgexport(gcf,'ex1_particle_distribution57_pre_resampling.eps',style);
    end
    if i==58
      hgexport(gcf,'ex1_particle_distribution58_pre_resampling.eps',style);
    end
    if i==59
      hgexport(gcf,'ex1_particle_distribution59_pre_resampling.eps',style);
    end
    if i==60
      hgexport(gcf,'ex1_particle_distribution60_pre_resampling.eps',style);
    end
    if i==61
      hgexport(gcf,'ex1_particle_distribution61_pre_resampling.eps',style);
    end
    if i==62
      hgexport(gcf,'ex1_particle_distribution62_pre_resampling.eps',style);
    end
    %plot the particle distributions after resampling
    figure(8);
    %hold on;
    plot(xk_1_2_pf, weightsk_1,'o');
    xlim([0 1]);
    ylim([0 0.05]);    
    xlabel('particle voltage (v)','FontSize',12);
    ylabel('particle weight','FontSize',12);
    title('Particle weight vs. particle voltage, tank 2, experiment #1, post-resampling');
    grid on; grid minor;
    %legendInfo = ['time index', num2str(i)];
    %legend(legendInfo)
    %legend('location', 'northeast');
    %saveas(gcf,'particle_PDF_%d','epsc',i)
    %print(figure(7), '-depsc', sprintf('particle_weights%d.eps', i));
    style = hgexport('factorystyle');
    style.Color = 'gray';
    if i==55
      hgexport(gcf,'ex1_particle_distribution55_post_resampling.eps',style);
    end
    if i==56
      hgexport(gcf,'ex1_particle_distribution56_post_resampling.eps',style);
    end
    if i==57
      hgexport(gcf,'ex1_particle_distribution57_post_resampling.eps',style);
    end
    if i==58
      hgexport(gcf,'ex1_particle_distribution58_post_resampling.eps',style);
    end
    if i==59
      hgexport(gcf,'ex1_particle_distribution59_post_resampling.eps',style);
    end
    if i==60
      hgexport(gcf,'ex1_particle_distribution60_post_resampling.eps',style);
    end
    if i==61
      hgexport(gcf,'ex1_particle_distribution61_post_resampling.eps',style);
    end
    if i==62
      hgexport(gcf,'ex1_particle_distribution62_post_resampling.eps',style);
    end
    %}

    %{
    %locate the healthy or faulty particles tracking the state of tank 2   
    healthy_indexes=find(X_boolean_1_t2_pf(1,:)==1);
    faulty_indexes=find(X_boolean_1_t2_pf(2,:)==1);   
    figure(6);
    clf;
    hold on;
    if isempty(healthy_indexes)~=1
        stem(Xk(healthy_indexes),weightsk(healthy_indexes),'b'); %blue stem are healthy particles
    end   
    if isempty(faulty_indexes)~=1
        stem(Xk(faulty_indexes),weightsk(faulty_indexes),'r');   %red stem are unhealthy particles
    end    
    xlim([0 2]);
    ylim([0 0.03]);
    xlabel('particle health (boolean)','FontSize',12);
    ylabel('particle weight','FontSize',12);
    pause(0.1);
    hold off;        
    
    %these are the scatter plots of the booleans [10] or [01]    
    figure(5)
    clf
    %hold on
    scatter(X_boolean_t2_perturbed(1,:),X_boolean_t2_perturbed(2,:))
    xlim([-1 2])
    ylim([-1 2])
    rectangle('position',[0.25 -0.75 1.5 1.5],'LineWidth',2)
    axis([-1 2 -1 2])
    rectangle('position',[-0.75 0.25 1.5 1.5],'LineWidth',2)
    xlabel('Boolean vector first element','FontSize',12)
    ylabel('Boolean vector second element','FontSize',12)
    title('Particle Boolean vector, first element vs. second element');
    plot([-1 2],[-1 2],'--k','LineWidth',2)
    plot(1,0,'r+','MarkerFaceColor','r','linewidth',2)
    plot(0,1,'r+','MarkerFaceColor','r','linewidth',2)
    legendInfo = ['time index', num2str(i)];
    legend(legendInfo)
    legend('location', 'northeast');
    %style = hgexport('factorystyle');
    %style.Color = 'gray';
    %hgexport(gcf,'figure_3_3.eps',style);
    %print(figure(5), '-depsc', sprintf('boolean_scatter_plots%d.eps', i));
    %pause(0.01)
    %hold off   
    %}
    
    %{
    %plot the particles against the process
    figure(13)
    clf;
    subplot(3,1,1)   
    hold on
    plot(x1)
    plot(i*ones(1,M),xkm1_t1pf,'o')
    hold off
    %
    subplot(3,1,2)   
    hold on
    plot(x2)
    plot(i*ones(1,M),xkm1_t2pf,'o')
    hold off
    %
    subplot(3,1,3)
    hold on
    plot(x3)
    plot(i*ones(1,M),xkm1_t2pf,'o')
    hold off
    pause(0.01)
    %}
    
    %{
    %plot the steps of the estimate against the process values
    figure(14)
    clf
    subplot(3,1,1) 
    hold on
    stem(x2(i),1,'r')
    stem(X_est_2_pf(i),1,'g')
    stem(xkm1_t2pf,weightsk)    
    hold off
    xlim([0 1])
    %
    subplot(3,1,2) 
    hold on
    stem(x2(i),1,'r')
    stem(X_est_2_pf(i),1,'g')
    stem(xkm1_t2pf,weightsk)
    hold off
    xlim([0 1])
    %
    subplot(3,1,3) 
    hold on
    stem(x3(i),1,'r')
    stem(X_est_3_pf(i),1,'g')
    stem(xkm1_t2pf,weightsk)
    hold off
    xlim([0 1])
    pause(0.01)
    %}    
end
  
%% plot probability of failure
figure(20);
%figure('units','normalized','outerposition',[0 0 1 1])
plot(E_failure_2_abrupt_fault, 'o-');
grid on;              
grid minor;
axis([zero_time_start time_stop 0 1]);
xlabel('time(s)'), ylabel('probability of fault occurrence'); 
title('tank 2 probability of fault occurrence');
style = hgexport('factorystyle');
style.Color = 'gray';
%hgexport(gcf,'figure_3_2.eps',style);
%saveas(gcf,'figure_3_2.jpg');
%saveas(gcf,'figure_3_1_inaccurate_model.jpg');
%cmap = colormap('gray');
%imwrite(gcf,cmap, 'figure_3_1.jpg', 'jpeg');
%I = ind2gray(grayimage,cmap);
%imwrite(I,'imagename.jpg');
saveas(gcf,'figure_3_2.jpg');
contents = dir('figure_3_2.jpg') % or whatever the filename extension is
for k = 1:numel(contents)
  filename   = contents(k).name;
  rgbImg     = imread(filename);
  gsImg      = rgb2gray(rgbImg);
  [~,name,~] = fileparts(filename);
  gsFilename = sprintf('%s_gs.jpg', name);
  imwrite(gsImg,gsFilename);
end

%% plot all 4 figures together
figure(23);
%figure('units','normalized','outerposition',[0 0 1 1]) %make the plot full screen
%%{
subplot(3, 1, 1);
plot(1:1:N, x1, '--', 1:1:N,x_est1, ':', 1:1:N-1, X_est_1_pf, '-.');
axis([zero_time_start time_stop 0 1.6]);
grid on;
grid minor;
%xticks([0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200]);
%yticks([0 0.2 0.4 0.6 0.8 1.0 1.2]);
legend('experimentally measured voltage', 'modelled voltage', 'voltage tracked with tank 2 PF')
legend('location', 'northeast');
xlabel('time(s)'), ylabel('voltage (V)'); 
title ('tank 1 voltage vs. time');
%%}
subplot(3, 1, 2);
plot(1:1:N, x2, '--', 1:1:N, x_est2, ':', 1:1:N-1, X_est_2_pf, '-.');
axis([zero_time_start time_stop 0 1.6]);
grid on; grid minor;
%xlim([50 65])
%ylim([0.55 0.7])
%xticks([0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200]);
%yticks([0 0.2 0.4 0.6 0.8 1.0 1.2]);
legend('experimentally measured voltage', 'modelled voltage', 'voltage tracked with tank 2 PF')
legend('location', 'northeast');
xlabel('time(s)'), ylabel ('voltage (V)'); 
title ('tank 2 voltage vs. time');
%
%%{
subplot(3, 1, 3);
plot(1:1:N, x3, '--', 1:1:N, x_est3, ':', 1:1:N-1, X_est_3_pf, '-.');
axis([zero_time_start time_stop 0 1.6]);
grid on;
grid minor;
%%xticks([0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200]);
%yticks([0 0.2 0.4 0.6 0.8 1.0 1.2]);
legend('experimentally measured voltage', 'modelled voltage', 'voltage tracked with tank 2 PF');
legend('location', 'northeast');
xlabel('time(s)'), ylabel ('voltage (V)'); 
title ('tank 3 voltage vs. time');
%%}
%style = hgexport('factorystyle');
%style.Color = 'gray';
%style.format = 'jpeg';
%hgexport(gcf,'figure_3_3.jpg',style);
%saveas(gcf,'figure_3_3.jpg');
saveas(gcf,'figure_3_3.jpg');
%cmap = colormap('gray');
%imwrite(gcf,cmap, 'figure_3_1.jpg', 'jpeg');
%I = ind2gray(grayimage,cmap);
%imwrite(I,'imagename.jpg');
contents = dir('figure_3_3.jpg') % or whatever the filename extension is
for k = 1:numel(contents)
  filename   = contents(k).name;
  rgbImg     = imread(filename);
  gsImg      = rgb2gray(rgbImg);
  [~,name,~] = fileparts(filename);
  gsFilename = sprintf('%s_gs.jpg', name);
  imwrite(gsImg,gsFilename);
end
return; 
 
%{
figure(15)
image(Qin_particles-Qout_particles,'CDataMapping','scaled')
colorbar
colormap jet
caxis([-0.06 0.05])
figure(16)
image(Qout_particles,'CDataMapping','scaled')
colorbar
colormap jet
caxis([-0.06 0.05])
figure(17)
image(Error2,'CDataMapping','scaled')
colorbar
colormap jet
figure(18)
image(Error2,'CDataMapping','scaled')
colorbar
colormap jet
figure(19)
hold on
plot(Q1+Q2,'b')
plot(Q1+Q2-Qleak,'r')
plot(Qleak,'r')
hold off
%}
