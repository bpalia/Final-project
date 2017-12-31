close all
clearvars
clc

fnum = 0;
cf = 50;
removed = 200;
K = 8; % prediction

load model_B
Am = MboxJ.D;
Cm = MboxJ.C;
Bm = conv(MboxJ.B,Am);
S = 24;

load('ptstu94.mat') % Input
load('utempSla_9395.dat')
y = utempSla_9395(:,3);
u = ptstu94; 

y(24:24:end) = nan;
y = fillmissing(y,'linear');

startDay = 430;
testWeek = 20; 
modelWeek = 10;

hrsInYear = 365*24;

% Final project part C, recursive predicton without input
% Find segment to predict
yT = y((startDay*24 + (modelWeek + testWeek)*7*24 + 1 - removed):(startDay*24 + ...
                                    (modelWeek + 10 + testWeek)*7*24));
uT = u((startDay*24 + (modelWeek + testWeek)*7*24 + 1 - hrsInYear - removed):(startDay*24 + ...
                                    (modelWeek + 10 + testWeek)*7*24 - hrsInYear+K));                                
                                
yT(16:24:end) = nan;
yT = fillmissing(yT,'linear');
yT(11) = nan; % taking out the outlier
yT = fillmissing(yT,'linear');
fnum = fnum+1;
figure(fnum)
plot(yT); title('Convergence (400 samples) and Test set')
hold on
plot(uT)

% State space equation definition
a1_var = 1e-6;
a2_var = 1e-6;
a3_var = 0;
a4_var = 0;
a11_var = 0;
c1_var = 1e-5;
c24_var = 1e-5;
c25_var = 1e-5;
b0_var = 1e-6;
b1_var = 1e-6;
b2_var = 1e-6;
b3_var = 0;
b4_var = 0;
b11_var = 0;
ord = 15;
A = eye(ord);
% Re = diag(0*ones(1,ord)); % Hiden state noise covariance matrix
Re = diag([a1_var a2_var a3_var a4_var a11_var 0 c1_var c24_var c25_var b0_var b1_var b2_var b3_var b4_var b11_var]);
Rw = 25; % Observation variance
% usually C should be set here to, but in this case C is a function of time

% set initial values
Rxx_1 = 10e-6 * eye(ord); % initial variance
xtt_1 = [Am(2:5) Am(12) -1 Cm(2) Cm(25:26) Bm(1:5) Bm(12)]'; % initial state

y = yT;
u = uT;

N = length(y);
e = zeros(1,N+K);
yhat = zeros(K,N);
% vector to store values in
xsave = zeros(ord,N);

% Kalman filter. Start from k=27, because we need old values of y
for n = 36:N
    % C is, in our case, a function of time
    yt = y(n);
    ut = u(n);
    Ct = [-y(n-1)+y(n-1-S), -y(n-2)+y(n-2-S), -y(n-3)+y(n-3-S), -y(n-4)+y(n-4-S), -y(n-11)+y(n-11-S), -y(n-S),...
            e(n-1), e(n-24), e(n-25),...
            u(n)-u(n-S), u(n-1)-u(n-1-S), u(n-2)-u(n-2-S), u(n-3)-u(n-3-S), u(n-4)-u(n-4-S), u(n-11)-u(n-11-S)];
    e(n) = yt-Ct*xtt_1;
    
    % Update
    Ryy = Ct*Rxx_1*Ct' + Rw;
    Kt = Rxx_1*Ct'/Ryy;
    xtt = xtt_1+Kt*(yt-Ct*xtt_1);
    Rxx = (eye(ord)-Kt*Ct)*Rxx_1;
    
    % Prediction
    for k = 1:K
        if k == 1
            Ck = [-y(n-1+k)+y(n-1-S+k), -y(n-2+k)+y(n-2-S+k), -y(n-3+k)+y(n-3-S+k), -y(n-4+k)+y(n-4-S+k), -y(n-11+k)+y(n-11-S+k), -y(n-S+k),...
                e(n-1+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k), u(n-4+k)-u(n-4-S+k),u(n-11+k)-u(n-11-S+k)];
            yhat(k,n) = Ck*xtt_1;
        elseif k == 2
            Ck = [-yhat(k-1,n)+y(n-1-S+k), -y(n-2+k)+y(n-2-S+k), -y(n-3+k)+y(n-3-S+k), -y(n-4+k)+y(n-4-S+k), -y(n-11+k)+y(n-11-S+k), -y(n-S+k),...
                e(n-1+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k), u(n-4+k)-u(n-4-S+k),u(n-11+k)-u(n-11-S+k)];
            yhat(k,n) = Ck*xtt_1;
        elseif k == 3
            Ck = [-yhat(k-1,n)+y(n-1-S+k), -yhat(k-2,n)+y(n-2-S+k), -y(n-3+k)+y(n-3-S+k), -y(n-4+k)+y(n-4-S+k), -y(n-11+k)+y(n-11-S+k), -y(n-S+k),...
                e(n-1+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k), u(n-4+k)-u(n-4-S+k),u(n-11+k)-u(n-11-S+k)];
            yhat(k,n) = Ck*xtt_1;  
        elseif k == 4
            Ck = [-yhat(k-1,n)+y(n-1-S+k), -yhat(k-2,n)+y(n-2-S+k), -yhat(k-3,n)+y(n-3-S+k), -y(n-4+k)+y(n-4-S+k), -y(n-11+k)+y(n-11-S+k), -y(n-S+k),...
                e(n-1+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k), u(n-4+k)-u(n-4-S+k),u(n-11+k)-u(n-11-S+k)];
            yhat(k,n) = Ck*xtt_1;            
        else
            Ck = [-yhat(k-1,n)+y(n-1-S+k), -yhat(k-2,n)+y(n-2-S+k), -yhat(k-3,n)+y(n-3-S+k), -yhat(k-4,n)+y(n-4-S+k), -y(n-11+k)+y(n-11-S+k), -y(n-S+k),...
                e(n-1+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k), u(n-4+k)-u(n-4-S+k),u(n-11+k)-u(n-11-S+k)];
            yhat(k,n) = Ck*xtt_1;            
        end
    end    
    
    % Save
    xsave(:,n) = xtt_1;
    
    % Predict
    Rxx_1 = A*Rxx*A'+Re;
    xtt_1 = A*xtt;  
end

%% Plot parameter estimates
fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(1:6,:)')
title('a parameters')
legend('a_1 Kalman','a_2 Kalman','a_3 Kalman','a_4 Kalman','a_{11} Kalman','a_{24}','location','southeast')

fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(7:9,:)')
title('c parameters')
legend('c_1 Kalman','c_{24} Kalman','c_{25}','location','southeast')

fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(10:15,:)')
title('b parameters')
legend('b_0 Kalman','b_1 Kalman','b_2 Kalman','b_3 Kalman','b_4 Kalman','b_{11}','location','southeast')

%% Plot modeling error
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
fnum = fnum +1;
figure(fnum)
time = 1:length(yT);
plot(time,yT,time+8,uT(K+1:end),time+8,yhat(8,:))
legend('output','input','prediction')
%%
err1stepCinput_mod_var = err_1_var;
err8stepCinput_mod_var = err_8_var;
save('test_Cinput_var','err1stepCinput_mod_var','err8stepCinput_mod_var')