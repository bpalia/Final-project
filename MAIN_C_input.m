close all
clearvars
clc

fnum = 0;
cf = 50;
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

startday = 430;
modelweek = 10;
predWeeks = 10;
hrsInYear = 24 * 365;

yM = y(startday*24+1:startday*24+modelweek*7*24);
uM = u((startday*24+1 - hrsInYear):(startday*24+modelweek*7*24 - hrsInYear));
yM(519) = nan; % taking out the outlier
yM = fillmissing(yM,'linear');

fnum = fnum+1;
figure(fnum)
plot(yM)

yValid = y((startday*24 + modelweek*24*7 + 1): (startday*24 + (modelweek + predWeeks)*24*7));
uValid = u(((startday*24 + modelweek*7*24) + 1 - hrsInYear): ((startday*24+(modelweek + predWeeks)*7*24) + 1 - hrsInYear)+K);
yFull = [yM; yValid];
uFull = [uM; uValid];

fnum = fnum+1;
figure(fnum)
plot(yFull); title('Modelling and validation set')
hold on
plot(uFull)

rm = 50;
yFullold = yFull;

% Defining all variances
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
Rw = 10; % Observation variance
% usually C should be set here to, but in this case C is a function of time

% set initial values
Rxx_1 = 10e-6 * eye(ord); % initial variance
xtt_1 = [Am(2:5) Am(12) -1 Cm(2) Cm(25:26) Bm(1:5) Bm(12)]'; % initial state

y = yFull;
u = uFull;

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
time = (1:N)/(24*7);
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
subplot(311);plot(time,xsave(7,:)'); ylabel('c_1 value'); ylim([-.85 -.75])
subplot(312);plot(time,xsave(8,:)'); ylabel('c_{24} value'); ylim([-.9 -.8])
subplot(313);plot(time,xsave(9,:)'); ylabel('c_{25} value'); ylim([.65 .75])
xlabel('Time [Weeks]'); 
set(gcf, 'Units','centimeters','Position',[0 0 18 10],'Units','centimeters', 'PaperSize',[18 10])

fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(10:15,:)')
title('b parameters')
legend('b_0 Kalman','b_1 Kalman','b_2 Kalman','b_3 Kalman','b_4 Kalman','b_{11}','location','southeast')

%% Plot modeling error
resid = e(modelweek*24*7 + 1:end);
fnum = func_plotacfpacf(fnum, resid, cf, 0.05, 'recursive estimation residuals');
fnum = fnum +1;
figure(fnum)
disp('WhitenessTest for residuals')
whitenessTest(resid)
title('Cumulative periodogram for resid')
fnum = fnum +1;
figure(fnum)
normplot(resid)
title('Norplot for resid')
%% k = 1 prediction
kk = 1;
yhat_1 = yhat(kk,modelweek*24*7 + 1-kk:end-kk);
err_1 = yValid - yhat_1';
err_1_var = var(err_1);
time = (1:length(yValid))/(24*7);
fnum = fnum +1;
figure(fnum)
plot(time,yValid,time,yhat_1)
title([num2str(kk), '-step prediction'])
xlabel('Time [Weeks]');ylabel('Temperature [^oC]');
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

fnum = fnum + 1;
figure(fnum)
acf(err_1, cf, 0.05, true, 0, 0);
title(['ACF of ' num2str(kk) '-step prediction errors'])
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

fnum = fnum +1;
figure(fnum)
disp('WhitenessTest for 1-step prediction')
whitenessTest(err_1)
title('Cumulative periodogram for error k=1')
fnum = fnum +1;
figure(fnum)
normplot(err_1)
title('Normplot for error k=1')

%% k = 8 prediction
kk = 8;
yhat_8 = yhat(kk,modelweek*24*7 + 1-kk:end-kk);
% yValid = yValid(100+1:end);
err_8 = yValid - yhat_8';
err_8_var = var(err_8);
fnum = fnum +1;
figure(fnum)
plot(time,yValid,time,yhat_8)
title([num2str(kk), '-step prediction'])
xlabel('Time [Weeks]');ylabel('Temperature [^oC]');
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

fnum = fnum + 1;
figure(fnum)
acf(err_8, cf, 0.05, true, 0, 0);
title(['ACF of ' num2str(kk) '-step prediction errors'])
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])
%%
err1step_Cinput_var = err_1_var;
err8step_Cinput_var = err_8_var;
save('variances_valid_Cinput', 'err1step_Cinput_var', 'err8step_Cinput_var')