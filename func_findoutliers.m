function [indicies] = func_findoutliers(y, alpha)
    N  = length(y);
    ys = sort( y );

    % Form trimmed data.
    g = N*alpha;
    indY = find( y < ys( floor(g)+1 ) );
    y(indY) = 0;
    indY = find( y > ys( floor(N-g+1)) );
    y(indY) = 0;

    % Find indicies 
    La = find( (y > ys( floor(g)+1 )) & (y < ys( floor(N-g+1))) );
    indicies = zeros(N,1);
    indicies(La) = 1;
end