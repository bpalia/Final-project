function [F,G] = func_poldiv(A,C,k)
[A,C] = equalLength(A,C);
[F,G] = deconv(conv([1 zeros(1,k-1)],C),A);
end

