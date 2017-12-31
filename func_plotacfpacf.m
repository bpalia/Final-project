function [fnum] = func_plotacfpacf(fnum, x, cutoff, alpha, tit)
    fnum = fnum + 1;
    figure(fnum)
    
    subplot(211)
    acf(x, cutoff, alpha, true, 0, 0);
    title(['ACF of ', tit])
    subplot(212)
    pacf(x, cutoff, alpha, true, 0);
    title(['PACF of ', tit])
    set(gcf, 'Units','centimeters','Position',[0 0 18 10],'Units','centimeters', 'PaperSize',[18 12])
end