% Project part B
clearvars
clc
close all
fnum = 0;
cf = 50;
hrsInYear = 24 * 365;
startday = 430;
modelweek = 10;
predWeeks = 10;
%% MOdeling
load('ptstu94.mat') % Input
load('utempSla_9395.dat') % Data

u = ptstu94; 
y = utempSla_9395(:,3);

% Interpolate for missing value
y(24:24:end) = nan;
y = fillmissing(y,'linear');

yM = y(startday*24+1:startday*24+modelweek*7*24);
yM(519) = nan; % taking out the outlier
yM = fillmissing(yM,'linear');

uM = u((startday*24+1 - hrsInYear):(startday*24+modelweek*7*24 - hrsInYear));

timeM = (1:length(uM))/(24*7);

% Nothing obvious strange going on
fnum = fnum + 1;
figure(fnum)
plot(timeM, uM)
title('SMHI temperature data in Sturup (1994)')
xlabel('Time [Weeks]'); ylabel('Temperature [^oC]')
set(gcf, 'Units','centimeters','Position',[0 0 18 7],'Units','centimeters', 'PaperSize',[18 7])

% Removing outliers in input data
removePercentage = 0.02;
[indicies] = func_findoutliers(uM, removePercentage);
uM(find(indicies - 1)) = nan;
uM = fillmissing(uM, 'linear');


fnum = fnum + 1;
figure(fnum)
plot(timeM, uM, timeM, yM)
title('SMHI temperature data for input (blue) and output(orange) in 1994')

% Removing mean values - Both tests suggest removal
disp('Test mean of uM before deseasoning')
testMean(uM, 0.05)
muM = mean(uM);
uM = uM - muM;

disp('Test mean of yM before deseasoning')
testMean(yM, 0.05)
myM = mean(yM);
yM = yM - myM;

% Normplot of uM suggests some transformation is required
fnum = fnum + 1;
figure(fnum)
normplot(uM)
title('Normality of SMHI prediction')

%% Removal of season
fnum = func_plotacfpacf(fnum, uM, cf, 0.05, 'of SMHI input before deseasoning');
A24 = [1, zeros(1,23) -1];

% Save both u, y before deseasoning (Needed?)
uNonD = uM;
yNonD = yM;

rm = 50;
uM = filter(A24, 1, uM);
uM(1:rm) = [];

fnum = func_plotacfpacf(fnum, uM, cf, 0.05, 'of SMHI input after deseasoning');

% Checking normality after deseasoning
fnum = fnum + 1;
figure(fnum)
normplot(uM)
title('Normality of SMHI prediction after deseasoning')

disp('Test mean of uM after deseasoning')
testMean(uM, 0.05)

%% Finding model orders for B and A2

data_uM = iddata(uM);

%% prewhitening ARMA(25,25)
model_init = idpoly([1, zeros(1,25)], [], [1, zeros(1,25)]);
model_init.Structure.a.Free = [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, zeros(1,13), 0, 1, 1];
model_init.Structure.c.Free = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, zeros(1,4), 0, 0, 0, 0, 1, 1];
arma_model = pem(data_uM, model_init);

res = resid(arma_model, data_uM);
fnum = func_plotacfpacf(fnum, res.y, cf, 0.05, 'ARMA(25,25) model residuals');
%   a_1, a_3, a_4, a_{24}, a_{25}, c_1, c_{24}, c_{25}
present(arma_model)

best_prew_model = arma_model;

%% Prewhiten
% Create prewhitened input and output
A_star = conv(A24, best_prew_model.a);

uM_pw = filter(A_star, best_prew_model.c, uNonD);
uM_pw(1:rm) = [];

yM_pw = filter(A_star, best_prew_model.c, yNonD);
yM_pw(1:rm) = [];

%% Checking crosscorrelation function between prewhitened x and y

fnum = func_plotccf(fnum, uM_pw, yM_pw, cf/2, 'u^{pw} and y^{pw}');
set(gcf, 'Units','centimeters','Position',[0 0 18 7],'Units','centimeters', 'PaperSize',[18 7])
ylabel('Amplitude')
% d = r = s = 0 

%% Estimate B and A2
d = 0; r = 0; s = 0;
A2 = [1, zeros(1, r)]; % r
B = [zeros(1, d), 0, zeros(1,s)]; % d, s
Mi = idpoly(1,B,[],[],A2);
Mi.Structure.b.Free = [zeros(1, d), 1, ones(1,s)]; % d, s

zpw = iddata(yM_pw,uM_pw);
Mba2 = pem(zpw,Mi); present(Mba2);
vhat = resid(Mba2,zpw);

% [verify that this is white?]
fnum = func_plotccf(fnum, uM_pw, vhat.y, cf/2, ['u^{pw} and v if d=', num2str(d), ', r=', num2str(r), ...
                                            ', s=', num2str(s)]);
set(gcf, 'Units','centimeters','Position',[0 0 18 7],'Units','centimeters', 'PaperSize',[18 7])
ylabel('Amplitude')
%% Deseason y
yM = filter(A24, 1, yM);
yM(1:rm) = [];

disp('Test mean of yM after deseasoning')
testMean(yM, 0.05)

h = filter(Mba2.b, Mba2.f, uM);
x = yM(51:end) - h(51:end);

%% Estimating orders for C1(z) A1(z)
fnum = func_plotccf(fnum, uNonD, x, cf, 'between u and x');

% Determine suitable model orders for C1, A1
fnum = func_plotacfpacf(fnum, x, cf, 0.05, 'x');

data_x = iddata(x);

%% ARMA(11,25)
model_init = idpoly([1, zeros(1, 11)], [], [1, zeros(1,25)]);
model_init.Structure.c.Free = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, zeros(1, 11), 1, 1];
model_init.Structure.a.Free = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1];
arma_model = pem(data_x, model_init);

res3 = resid(arma_model, data_x);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'ARMA(11, 25) residuals');
%  a_1 a_4, a_{11}, c_1, c_{24}, c_{25}
present(arma_model)

best_c1a1_model = arma_model;

%% Final model TEST - (d,r,s) = (0,0,0), C1 - 25, A1 - 11, A2 - 0

A1 = [1, zeros(1,11)]; 
A2 = [1];
B = [1];
C = [1 zeros(1,25)];
Mi = idpoly(1, B, C, A1, A2);

Mi.Structure.c.Free = [0, 1, zeros(1,22), 1, 1];
Mi.Structure.d.Free = [0, 1, 1, 1, 1, zeros(1, 6), 1];
z = iddata(yM,uM);
MboxJ = pem(z,Mi);

present(MboxJ);
ehat = resid(MboxJ,z);

disp('%%%%%%%% Final model - (d,r,s) = (0,0,0), C1 - 25, A1 - 11, A2 - 0 %%%%%%%%%%')

fnum = fnum + 1;
figure(fnum)
whitenessTest(ehat.y, 0.05)
title('Cumulative periodogram for input-output model')

fnum = func_plotacfpacf(fnum, ehat.y, cf, 0.05, 'final model residuals');

%% Predictions
C = MboxJ.c;
A1 = MboxJ.d;
A2 = MboxJ.f;
B = MboxJ.b;

A = conv(A1, A2);
A = conv(A, A24);
C = conv(C, A2);
B = conv(B, A1);
B = conv(B, A24);

%% k = 1
k = 1;

uValid = u(((startday*24 + modelweek*7*24) + 1 - k - hrsInYear): ((startday*24+(modelweek + predWeeks)*7*24) + 1 - hrsInYear));
yValid = y(((startday*24 + modelweek*7*24) + 1 - k): ((startday*24 + (modelweek + predWeeks)*7*24) + 1));

[F,G] = func_poldiv(A,C,k);
BF = conv(B,F);
[Fhat,Ghat] = func_poldiv(C,BF,k);

yhat = filter(Ghat,C,uValid) + filter(G,C,yValid) + filter(Fhat,1,uValid);
yhat = yhat(k+2:end);
yValid = yValid(k+2:end);

timeV = 3*(1:length(yhat))/(24*7);

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
acf(err1step, cf, 0.05, true, 0, 0)
title(['ACF of ' num2str(k) '-step prediction errors'])
set(gcf, 'Units','centimeters','Position',[0 0 12 10],'Units','centimeters', 'PaperSize',[12 10])

fnum = fnum + 1;
figure(fnum)
whitenessTest(err1step)

%% k = 8
k = 8;

uValid = u(((startday*24+modelweek*7*24) + 1 - k - hrsInYear): ((startday*24+(modelweek + predWeeks)*7*24) + 1 - hrsInYear));
yValid = y(((startday*24+modelweek*7*24) + 1 - k): ((startday*24+(modelweek + predWeeks)*7*24) + 1));

[F,G] = func_poldiv(A,C,k);
BF = conv(B,F);
[Fhat,Ghat] = func_poldiv(C,BF,k);

skip = max([length(Ghat), length(Fhat)]);

yhat = filter(Ghat,C,uValid) + filter(G,C,yValid) + filter(Fhat,1,uValid);
yhat = yhat(skip:end);
yValid = yValid(skip:end);

timeV = 3*(1:length(yhat))/(24*7);

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

%% Save results

err1step_B_var = err1step_var;
err8step_B_var = err8step_var;

save('Model_B', 'MboxJ')
save('variances_valid_B', 'err1step_B_var', 'err8step_B_var')


