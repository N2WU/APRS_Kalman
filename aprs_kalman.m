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
data_cutoff = 10; %how many to use as training data
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
A = [1 T; 0 1]; B = [T^2/2 T]; H = [1 0];
var_alt = var(alt_train);

n = 1;
x(n) = time_est(n);
yv(n) = yv(n-1)+ya(n-1)*T;
yp(n) = yp(n-1) + yv(n-1)*T + 1/2*ya(n-1)*T^2;
eta(n) = ya(n-1);
y(n) = [yp(n); yv(n)];

%% Predict and Format
% use hw7
for n=1:1e3
    y_est(n) = A*y_est(n-1);
    Ry(n) = A*Ry(n-1)*A' + B*Rn*B'
    
    K(n) = Ry(n)*H'*inv(Rw(n))
    y_est(n) = y_est(n)+K(n)*(x(n)-x_est(n))
    Ry = (I)
end