function fnum = func_plotccf(fnum, x, y, cutoff, tit)
    fnum = fnum +1;
    figure(fnum)
    stem(-cutoff:cutoff,crosscorr(x,y,cutoff));
    title(['Crosscorrelation function between ', tit]); 
    xlabel('Lag');
    hold on
    plot(-cutoff:cutoff,2/sqrt(length(x))*ones(1,2*cutoff+1),'--')
    plot(-cutoff:cutoff,-2/sqrt(length(x))*ones(1,2*cutoff+1),'--')
end