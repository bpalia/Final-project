close all
clearvars
clc

fnum = 0;
cf = 50;

load model_A
Am = best_model.A;
Cm = best_model.C;
S = 24;

load('utempSla_9395.dat')
y = utempSla_9395(:,3);

y(24:24:end) = nan;
y = fillmissing(y,'linear');

startday = 430;
modelweek = 10;
predWeeks = 10;

yM = y(startday*24+1:startday*24+modelweek*7*24);
yM(519) = nan; % taking out the outlier
yM = fillmissing(yM,'linear');

% Removing outliers in input data
[indicies] = func_findoutliers(yM, 0.02);
yM(find(indicies - 1)) = nan;
yM = fillmissing(yM, 'linear');

fnum = fnum+1;
figure(fnum)
plot(yM)

yValid = y((startday*24 + modelweek*24*7 + 1): (startday*24 + (modelweek + predWeeks)*24*7));

yFull = [yM; yValid];

fnum = fnum+1;
figure(fnum)
plot(yFull); title('Modelling and validation set')

rm = 50;

%%
y = yFull;
N = length(y);
% State space equation definition
ord = 4;
A = eye(ord);
% Re = diag([10e-4 10e-4 0 10e-4]); % Hiden state noise covariance matrix
Re = diag([0 0 0 5*10e-6]);
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
    Ct = [-y(n-1)+y(n-1-S) -y(n-2)+y(n-2-S) -y(n-S) e(n-24)];
    e(n) = yt-Ct*xtt_1;
    
    % Update
    Ryy = Ct*Rxx_1*Ct' + Rw;
    Kt = Rxx_1*Ct'/Ryy;
    xtt = xtt_1+Kt*(yt-Ct*xtt_1);
    Rxx = (eye(ord)-Kt*Ct)*Rxx_1;
    
    % Prediction
    for k = 1:K
        if k == 1
            Ck = [-yFull(n-1+k)+yFull(n-1-S+k) -yFull(n-2+k)+yFull(n-2-S+k) -yFull(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;
        elseif k == 2
            Ck = [-yhat(k-1,n)+yFull(n-1-S+k) -yFull(n-2+k)+yFull(n-2-S+k) -yFull(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;
        else
            Ck = [-yhat(k-1,n)+yFull(n-1-S+k) -yhat(k-2,n)+yFull(n-2-S+k) -yFull(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;            
        end
    end
    % Save
    xsave(:,n) = xtt_1;
    
    % Predict
    Rxx_1 = A*Rxx*A'+Re;
    xtt_1 = A*xtt;  
end

%%
time = (1:N)/(24*7);
fnum = fnum +1;
figure(fnum)
plot(time,xsave(:,:)')
legend('a_1 Kalman','a_2 Kalman','a_{24}','c_{24} Kalman','location','southeast')

fnum = fnum +1;
figure(fnum)
plot(time,xsave(4,:)'); ylim([-.9 -.7])
xlabel('Time [Weeks]'); ylabel('Value')
set(gcf, 'Units','centimeters','Position',[0 0 18 7],'Units','centimeters', 'PaperSize',[18 7])


resid = e(modelweek*24*7 + 1:end);
fnum = func_plotacfpacf(fnum, resid, cf, 0.05, 'recursive estimation residuals');
fnum = fnum +1;
figure(fnum)
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
whitenessTest(err_1)
title(['Cumulative periodogram for prediction errors k=', num2str(kk)])
fnum = fnum +1;
figure(fnum)
normplot(err_1)
title(['Normplot for prediction errors k=', num2str(kk)])

%% k = 8 prediction
kk = 8;
yhat_8 = yhat(kk,modelweek*24*7 + 1-kk:end-kk);
% yValid = yValid(100+1:end);
err_8 = yValid - yhat_8';
err_8_var = var(err_8);
fnum = fnum +1;
figure(fnum)
plot(time,yValid,time,yhat_8)
title([num2str(k), '-step prediction'])
xlabel('Time [Weeks]');ylabel('Temperature [^oC]');
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

fnum = fnum + 1;
figure(fnum)
acf(err_8, cf, 0.05, true, 0, 0);
title(['ACF of ' num2str(kk) '-step prediction errors'])
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

%%
err1step_C_var = err_1_var;
err8step_C_var = err_8_var;
save('variances_valid_C', 'err1step_C_var', 'err8step_C_var')