close all
clearvars
clc

fnum = 0;
cf = 50;
removed = 200;

load model_A
Am = best_model.A;
Cm = best_model.C;
S = 24;

% Load data
load('utempSla_9395.dat')
y = utempSla_9395(:,3);

startDay = 430;
testWeek = 20; 
modelWeek = 10;

hrsInYear = 365*24;

% Final project part C, recursive predicton without input
% Find segment to predict
yT = y((startDay*24 + (modelWeek + testWeek)*7*24 + 1 - removed):(startDay*24 + ...
                                    (modelWeek + 10 + testWeek)*7*24));


yT(8:24:end) = nan;
yT = fillmissing(yT,'linear');
yT(11) = nan; % taking out the outlier
yT = fillmissing(yT,'linear');
fnum = fnum+1;
figure(fnum)
plot(yT); title('Convergence (200 samples) and Test set')

%%
y = yT;
N = length(y)
% State space equation definition
ord = 4;
A = eye(ord);
Re = diag([0 0 0 5*10e-6]); % Hiden state noise covariance matrix
Rw = 25; % Observation variance
% usually C should be set here to, but in this case C is a function of time

% set initial values
Rxx_1 = 10e-5 * eye(ord); % initial variance
xtt_1 = [Am(2:end) -1 Cm(end)]'; % initial state

e = zeros(1,N);
% vector to store values in
xsave = zeros(ord,N);
K = 8; % prediction
yhat = zeros(K,N);

% Kalman filter. Start from k=27, because we need old values of y
for n = 27:N
    % C is, in our case, a function of time
    yt = y(n);
    Ct = [-y(n-1)+y(n-1-S), -y(n-2)+y(n-2-S), -y(n-S), e(n-24)];
    e(n) = yt-Ct*xtt_1;
    
    % Update
    Ryy = Ct*Rxx_1*Ct' + Rw;
    Kt = Rxx_1*Ct'/Ryy;
    xtt = xtt_1+Kt*(yt-Ct*xtt_1);
    Rxx = (eye(ord)-Kt*Ct)*Rxx_1;
    
    % Prediction
    for k = 1:K
        if k == 1
            Ck = [-y(n)+y(n-1-S+k) -y(n-1)+y(n-2-S+k) -y(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;
        elseif k == 2
            Ck = [-yhat(k-1,n)+y(n-1-S+k) -y(n)+y(n-2-S+k) -y(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;
        else
            Ck = [-yhat(k-1,n)+y(n-1-S+k) -yhat(k-2,n)+y(n-2-S+k) -y(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;            
        end
    end
    % Save
    xsave(:,n) = xtt_1;
    
    % Predict
    Rxx_1 = A*Rxx*A'+Re;
    xtt_1 = A*xtt;  
end

fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(:,:)')
legend('a_1 Kalman','a_2 Kalman','a_{24}','c_{24} Kalman','location','southeast')

resid = e(removed+1:end);
fnum = func_plotacfpacf(fnum, resid, cf, 0.05, 'resid');
fnum = fnum +1;
figure(fnum)
disp('%%%%%%%% RESID %%%%%%%%')
whitenessTest(resid)
title('Cumulative periodogram for resid')
fnum = fnum +1;
figure(fnum)
normplot(resid)
title('Norplot for resid')

%% k = 1 prediction
kk = 1;
yhat_1 = yhat(kk,removed+1-kk:end-kk);
yValid = yT(removed+1:end);
err_1 = yValid - yhat_1';
err_1_var = var(err_1);
time = 1:length(yValid);
fnum = fnum +1;
figure(fnum)
plot(time,yValid,time,yhat_1)

fnum = func_plotacfpacf(fnum, err_1, cf, 0.05, 'error k=1');
fnum = fnum +1;
figure(fnum)
disp('%%%%%%%% 1-STEP %%%%%%%%')
whitenessTest(err_1)
title('Cumulative periodogram for error k=1')
fnum = fnum +1;
figure(fnum)
normplot(err_1)
title('Normplot for error k=1')

%% k = 8 prediction
kk = 8;
yhat_8 = yhat(kk,removed+1-kk:end-kk);
err_8 = yValid - yhat_8';
err_8_var = var(err_8);
time = 1:length(yValid);
fnum = fnum +1;
figure(fnum)
plot(time,yValid,time,yhat_8)

fnum = func_plotacfpacf(fnum, err_8, cf, 0.05, 'error k=8');
fnum = fnum +1;
figure(fnum)
normplot(err_8)
title('Normplot for error k=8')
%%
err1stepC_mod_var = err_1_var;
err8stepC_mod_var = err_8_var;
save('test_C_var','err1stepC_mod_var','err8stepC_mod_var')