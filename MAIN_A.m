% part A
clearvars
clc
close all
fnum = 0;
cf = 50;
startday = 430;
modelweek = 10;
predWeeks = 10;

%% Modeling 
load('utempSla_9395.dat')
y = utempSla_9395(:,3);

y(24:24:end) = nan;
y = fillmissing(y,'linear');

yM = y(startday*24+1:startday*24+modelweek*7*24);
yM(519) = nan; % taking out the outlier
yM = fillmissing(yM,'linear');

time = (1:length(y))/(24*7);
timeM = (1:length(yM))/(24*7);

fnum = fnum + 1;
figure(fnum)
plot(time, y)
title('Temperature data in Svedala (1993-1995)')
xlabel('Time [Weeks]');ylabel('Temperature [^oC]'); xlim([0 126])
set(gcf, 'Units','centimeters','Position',[0 0 18 7],'Units','centimeters', 'PaperSize',[18 7])

fnum = fnum + 1;
figure(fnum)
plot(timeM, yM)
title('Temperature data in Svedala starting day 430')
xlabel('Time [Weeks]'); ylabel('Temperature [^oC]')
set(gcf, 'Units','centimeters','Position',[0 0 18 7],'Units','centimeters', 'PaperSize',[18 7])

% Normplot suggests some transformation is required
% Transformation, check Box Jenkins
fnum = fnum + 1;
figure(fnum)
bcNormPlot(yM);grid on; yticks(-6000:500:-2000); xticks(-2:0.5:2)
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])
% Suggests square root transformation, but data suggests keep yM

% Subtract mean
disp('testMean result before mean substraction and deseasoning')
testMean(yM, 0, 0.05)
myM = mean(yM);
yM = yM - myM;
yMnonD = yM;

fnum = func_plotacfpacf(fnum, yM, cf, 0.05, 'data before deseasoning');

% Shows strong seasonality, desason by 24
% move to ARMA process NOT!
A24 = [1, zeros(1,23) -1];

rm = 50;
yM = filter(A24, 1, yM);
yM(1:rm) = [];

fnum = func_plotacfpacf(fnum, yM, cf, 0.05, 'data after deseasoning');

% testMean() gives 1, need to substract
disp('testMean result after deseasoning')
testMean(yM, 0, 0.05)
myM = mean(yM);
yM = yM - myM;

% After deseasonalizing data appears t-distributed
fnum = fnum + 1;
figure(fnum)
normplot(yMnonD);title('Before deseasoning')
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])
fnum = fnum + 1;
figure(fnum)
normplot(yM);title('After deseasoning')
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

data_yM = iddata(yM);
%% AR(2)
ar_model2 = arx(data_yM, 2);
res = resid(ar_model2, data_yM);
fnum = func_plotacfpacf(fnum, res.y, cf, 0.05, 'AR(2) model residuals ');

%% ARMA(2,24)- a1, a2, c24
model_init = idpoly([1, 0, 0], [], [1, zeros(1,24)]);
model_init.Structure.c.Free = [zeros(1,24), 1];
arma_model = pem(data_yM, model_init);

res = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res.y, cf, 0.05, 'ARMA(2,24) model residuals ');

fnum = fnum +1;
figure(fnum)
disp(['Whiteness test for prediction residuals'])
whitenessTest(res.y)
title('Cumulative periodogram for residuals ARMA(2,24) a_1, a_2, c_{24}')
best_model = arma_model;
present(best_model)
%% Prediction
A = best_model.a;
C = best_model.c;

A_star = conv(A, A24);
%% k = 1
k = 1;

yValid = y((startday*24 + modelweek*24*7 + 1 - k): (startday*24 + (modelweek + predWeeks)*24*7));

[F,G] = func_poldiv(A_star,C,k);
yhat = filter(G,C,yValid);
yhat(1:k) = [];
yValid(1:k) = [];

timeV = (1:length(yValid))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeV, yValid, timeV, yhat)
title([num2str(k), '-step prediction'])
xlabel('Time [Weeks]');ylabel('Temperature [^oC]');
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

err1step = yValid - yhat;
err1step_var = var(err1step);

fnum = fnum + 1;
figure(fnum)
acf(err1step, cf, 0.05, true, 0, 0);
title(['ACF of ' num2str(k) '-step prediction errors'])
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

fnum = fnum + 1;
figure(fnum)
disp(['Whiteness test for prediction errors k=' num2str(k)])
whitenessTest(err1step)
title(['Cumulative periodogram for prediction errors k=' num2str(k)])

%% k = 8
k = 8;
filtermax = 24;
z = max(k,filtermax);

yValid = y((startday*24 + modelweek*24*7 + 1 - k): (startday*24 + (modelweek + predWeeks)*24*7));

[F,G] = func_poldiv(A_star,C,k);
yhat = filter(G,C,yValid);
yhat(1:z) = [];
yValid(1:z) = [];

timeV = (1:length(yValid))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeV, yValid, timeV, yhat)
title([num2str(k), '-step prediction'])
xlabel('Time [Weeks]');ylabel('Temperature [^oC]');
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

err8step = yValid - yhat;
err8step_var = var(err8step);

fnum = fnum + 1;
figure(fnum)
acf(err8step, cf, 0.05, true, 0, 0);
title(['ACF of ' num2str(k) '-step prediction errors'])
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

%% Save polynomials for A, C

err1step_A_var = err1step_var;
err8step_A_var = err8step_var;

save('Model_A', 'best_model')
save('variances_valid_A', 'err1step_A_var', 'err8step_A_var')






