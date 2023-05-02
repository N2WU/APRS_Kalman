clear all;
B = 8; Ss_den = [0.81,-1.629,4.9322,-1.629,.81];
B = B/Ss_den(1); Ss_den = Ss_den/Ss_den(1);
poles = roots(Ss_den);
a_in = poly(poles(3:4)); % Inside UC poles
a_out = poly(poles(1:2)); % Outside UC poles
b = sqrt(B);
a = a_in; N = 100;
eta = randn(N+1,1);
sn = filter(b,a,eta);
s_std = std(sn);
% Generate vector process y(n)
yn = sn; y_std = s_std;
% Generate the observation process x(n)
g1 = sqrt(20); v = g1*randn(N+1,1); %noise variance?
yn1 = [0;yn(1:N)];
h1 = 1.0; h2 = 0.0;
xn = h1*yn + h2*yn1 + v; % measurement value; xn = alt data
x_std = std(xn);
in_snr = 20*log10(y_std/g1);
% Design KF parameters
A = [-a(2),-a(3);1,0]; B = [b;0];
H = [h1,h2]; G = g1;
Rv = g1*g1; R_eta = eye(1);
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
error = yn(1:end-1) - y_hat(1:end-1); e_std = std(error);
out_snr = 20*log10(y_std/e_std);
% Plots
n = 0:N; figure;
plot(n,yn,'x',n,xn,':',n,y_hat,'g--','linewidth',1);
ylabel('Amplitude'); xlabel('Index n');
legend('y(n)','x(n)','yhat(n)'); axis([0,N,-15,15]);
set(gca,'xtick',(0:20:N),'ytick',(-15:5:15)); grid;
title('Estimation of AR(2) Process Using Kalman Filter');