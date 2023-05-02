%% APRS_kalman_2.m
% Loads an aprs.fi export and predicts next state as xy coordinates
close all; clear all;
%% Load Data
% Most of these variables are important; chiefly x and y, but also
% speed/time/altitude
datacsv = readtable('datasets/w2kgy-12_balloon.csv');
% let's start with just altitude - more linear
x_data = table2array(datacsv(:,"lng"));
y_data = table2array(datacsv(:,"lat"));
alt_data = table2array(datacsv(:,"altitude"));
alt_data = alt_data * 0.3048 /1e3; %convert to km
vel_data = table2array(datacsv(:,"speed"));
time_data = table2array(datacsv(:,"lasttime"));
% plot(alt_data) % this actually doesn't need much sanitization
data_cutoff = 31; %how many to use as training data
alt_train = alt_data(1:data_cutoff);
alt_est = alt_data(1+data_cutoff:end);
time_train = time_data(1:data_cutoff);
time_est = time_data(1+data_cutoff:end);

%% Estimate
% Calculate Kalman inputs A,B,Rn,H,Rv,x
T = mean(seconds(diff(time_train)));
A = [1 T; 0 1]; B = [0.5*(T^2); T];
N = data_cutoff-1;
% Generate the observation process x(n)
xn = alt_train-min(alt_train);
x_std = cov(xn)^2;
% Design KF parameters
H = [0 1];
Rv = cov(xn)^2; R_eta = eye(1);
% Initialization %xn(2)/T
y_post = [xn(1);xn(2)/T]; y_post = [0; 1]; 
R_post = zeros(2); %initial position and velocity known
IR = eye(size(R_post));
y_hat = zeros(N+1,1);
for n = 0:N
    % Predict
    R_pri = A*R_post*A' + B*R_eta*B';
    y_pri = A*y_post;
    x_pri = H*y_pri;
    % Update
    Rw = H*R_pri*H'+Rv;
    K = R_pri*H'*inv(Rw);
    %K = R_pri*H'\Rw; %*(Rw\IRv);
    y_post = y_pri + K*(xn(n+1) - x_pri);
    R_post = (IR-K*H)*R_pri;
    y_hat(n+1) = y_post(2);
end
err_avg = mean(xn(n+1)-x_pri);
for n=(N+1):111-1
    % Predict
    R_pri = A*R_post*A' + B*R_eta*B';
    y_pri = A*y_post;
    x_pri = H*y_pri;
    % Update
    Rw = H*R_pri*H'+Rv;
    K = R_pri*H'*inv(Rw);
    %K = R_pri*H'\Rw; %*(Rw\IRv);
    y_post = y_pri + K*(err_avg);
    R_post = (IR-K*H)*R_pri;
    y_hat(n+1) = y_post(2);
end
y_hat = [y_hat(2:end);y_hat(end)];
n = 0:110; figure;
plot(n,alt_data(n+1),':',n,y_hat,'g--','linewidth',1);
ylabel('Elevation (km)'); xlabel('Index n');
xline(data_cutoff)
legend('x(n)','yhat(n)','training data cutoff'); axis([0,length(n),0,max(xn)+10]);
set(gca,'xtick',(0:20:length(n)),'ytick',(0:5:max(xn)+10)); grid;
title('Estimation of Altitude Using Kalman Filter');