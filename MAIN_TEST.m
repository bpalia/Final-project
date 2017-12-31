close all
clearvars
clc

% Final project part C
% The three models in A/B/C are compared

removed = 25; % Samples to be removed to get prediction going
fnum = 0;
cf = 30;

% Load models
load('Model_A.mat')
load('Model_B.mat')

model_A = best_model;
model_B = MboxJ;

% Load validation variances
load('variances_valid_A.mat')
load('variances_valid_B.mat')
load('variances_valid_C.mat')
load('variances_valid_Cinput.mat')
load('test_C_var.mat')
load('test_Cinput_var.mat')


% Load data
load('utempSla_9395.dat')
load('ptstu94.mat')

y = utempSla_9395(:,3);
u = ptstu94;

startDay = 430;
testWeek = 20; 
modelWeek = 10;

hrsInYear = 365*24;

% Find segment to predict
yModel = y((startDay*24 + (modelWeek + testWeek)*7*24 + 1 - removed):(startDay*24 + ...
                                    (modelWeek + 10 + testWeek)*7*24));
uModel = u((startDay*24 + (modelWeek + testWeek)*7*24 + 1 - hrsInYear - removed):(startDay*24 + ...
                                    (modelWeek + 10 + testWeek)*7*24 - hrsInYear));

% Remove erronous samples in yModel
yModel(1:24:end) = nan;
yModel = fillmissing(yModel, 'linear');

timeM = (1:length(uModel))/(24*7);

fnum = fnum + 1;
figure(fnum)
plot(timeM, yModel)
hold on
plot(timeM, uModel)
xlabel('Week')
ylabel('Temperature')
title('Temperature and input for modeling period')
legend('Svedala', 'Sturup')

%% ############## MODEL A ##############
mod = 'A';

A = model_A.a;
C = model_A.c;

A24 = [1, zeros(1,23), -1];
A_star = conv(A, A24);

%% A, k = 1
k = 1;

yM = yModel;

[F,G] = func_poldiv(A_star, C, k);
yhat = filter(G, C, yM);
yhat(1:removed) = [];
yM(1:removed) = [];

timeM = (1:length(yM))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeM, yM)
hold on
plot(timeM, yhat)
title(['Model ', mod, ' ',num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')
legend('Measured', 'Predicted')

err1step = yM - yhat;
err1stepA_mod_var = var(err1step);

fnum = fnum + 1;
figure(fnum)
acf(err1step, cf, 0.05, true, 0, 0);
title(['Model ', mod, ', Residuals prediction k=', num2str(k)])

%% A, k = 8
k = 8;

yM = yModel;

[F,G] = func_poldiv(A_star, C, k);
yhat = filter(G, C, yM);
yhat(1:removed) = [];
yM(1:removed) = [];

timeM = (1:length(yM))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeM, yM)
hold on
plot(timeM, yhat)
title(['Model ', mod, ' ',num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')
legend('Measured', 'Predicted')

err1step = yM - yhat;
err8stepA_mod_var = var(err1step);

fnum = fnum + 1;
figure(fnum)
acf(err1step, cf, 0.05, true, 0, 0);
title(['Model ', mod, ', Residuals prediction k=', num2str(k)])

%% ############## MODEL B ##############

mod = 'B';

C = model_B.c;
A1 = model_B.d;
A2 = model_B.f;
B = model_B.b;

A = conv(A1, A2);
A = conv(A, A24);
C = conv(C, A2);
B = conv(B, A1);
B = conv(B, A24);

%% Model B, k = 1
k = 1;

yM = yModel;
uM = uModel;

[F,G] = func_poldiv(A,C,k);
BF = conv(B,F);
[Fhat,Ghat] = func_poldiv(C,BF,k);

yhat = filter(Ghat,C,uM) + filter(G,C,yM) + filter(Fhat,1,uM);
yhat = yhat(removed:end);
yM = yM(removed:end);

timeM = (1:length(yhat))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeM, yM)
hold on
plot(timeM, yhat)
title(['Model ', mod, ' ',num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')
legend('Measured', 'Predicted')

err1step = yM - yhat;
err1stepB_mod_var = var(err1step);

fnum = fnum + 1;
figure(fnum)
acf(err1step, cf, 0.05, true, 0, 0);
title(['Model ', mod, ', Residuals prediction k=', num2str(k)])

%% Model B, k = 8
k = 8;

yM = yModel;
uM = uModel;

[F,G] = func_poldiv(A,C,k);
BF = conv(B,F);
[Fhat,Ghat] = func_poldiv(C,BF,k);

yhat = filter(Ghat,C,uM) + filter(G,C,yM) + filter(Fhat,1,uM);
yhat = yhat(removed:end);
yM = yM(removed:end);

timeM = (1:length(yhat))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeM, yM)
hold on
plot(timeM, yhat)
title(['Model ', mod, ' ',num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')
legend('Measured', 'Predicted')

err1step = yM - yhat;
err8stepB_mod_var = var(err1step);

fnum = fnum + 1;
figure(fnum)
acf(err1step, cf, 0.05, true, 0, 0);
title(['Model ', mod, ', Residuals prediction k=', num2str(k)])

%% Plot results
barWidth = 0.4;

fnum = fnum + 1;
figure(fnum)
c = categorical({'Model A', 'Model B', 'Model A recursive', 'Model B recursive'});
bar(c, [err1stepA_mod_var, err1stepB_mod_var, err1stepC_mod_var, err1stepCinput_mod_var], ...
    'FaceColor', [0.8500    0.3250    0.0980], 'BarWidth', barWidth)
grid on
title('Comparison error 1 step predictions')
ylabel('Error')

fnum = fnum + 1;
figure(fnum)
bar(c, [err8stepA_mod_var, err8stepB_mod_var, err8stepC_mod_var, err8stepCinput_mod_var], ...
    'FaceColor', [0.8500    0.3250    0.0980], 'BarWidth', barWidth)
grid on
title('Comparison error 8 step predictions')
ylabel('Error')














