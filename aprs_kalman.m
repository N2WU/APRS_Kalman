%% APRS_kalman.m
% Loads an aprs.fi export and predicts next state as xy coordinates
close all;
%% Load Data
% Most of these variables are important; chiefly x and y, but also
% speed/time/altitude
datacsv = readtable('datasets/w2kgy-12_balloon.csv');
% let's start with just altitude - more linear
x_data = table2array(datacsv(:,"lng"));
y_data = table2array(datacsv(:,"lat"));
alt_data = table2array(datacsv(:,"altitude"));
alt_data = alt_data/1e3; %convert to kft
vel_data = table2array(datacsv(:,"speed"));
time_data = table2array(datacsv(:,"lasttime"));
% plot(alt_data) % this actually doesn't need much sanitization
data_cutoff = 51; %how many to use as training data
alt_train = alt_data(1:data_cutoff);
alt_est = alt_data(1+data_cutoff:end);
time_train = time_data(1:data_cutoff);
time_est = time_data(1+data_cutoff:end);

%% Estimate
% Calculate Kalman inputs A,B,Rn,H,Rv,x
time_diff = zeros(1,length(time_train)-1);
for k=1:length(time_diff)
    time_diff(k) = second(time_train(k)) -second(time_train(k+1));
end
T = mean(time_diff);
A = [1 T; 0 1]; B = [T^2/2 T];
var_alt = var(alt_train);
%% Predict and Format

% Generate vector process y(n)
N = 50;
% Generate the observation process x(n)
g1 = 0.25;
xn = alt_train;
x_std = std(xn);
% Design KF parameters
H = [1 0]; G = 0.25;
Rv = 0.25^2; R_eta = eye(1);
% Initialization
y_post = [0;1]; R_post = zeros(2);
IRv = eye(size(Rv)); IR = eye(size(R_post));
y_hat = zeros(N+1,1); gain = zeros(N+1,2); mse = y_hat;
% Tracking
for n = 0:N
R_pri = A*R_post*A' + B*R_eta*B';
y_pri = A*y_post;
x_pri = H*y_pri;
Rw = H*R_pri*H'+Rv;
K = R_pri*H'*(Rw\IRv);
y_post = y_pri + K*(xn(n+1) - x_pri);
R_post = (IR-K*H)*R_pri;
y_hat(n+1) = y_post(2);
gain(n+1,:) = K';
mse(n+1) = R_post(1,1);
end
y_hat = [y_hat(2:N+1);y_hat(N+1)];

n = 0:N; figure;
plot(n,xn,':',n,y_hat,'g--','linewidth',1);
ylabel('Amplitude'); xlabel('Index n');
legend('x(n)','yhat(n)'); axis([0,N,0,max(xn)+10]);
set(gca,'xtick',(0:20:N),'ytick',(0:5:max(xn)+10)); grid;
title('Estimation of Altitude Using Kalman Filter');