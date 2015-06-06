% Work out what F Diag F' is for F = DCT matrix
clear all

N = 8; F = dctmtx(N);
% python code does it as below
% scale = sqrt(2/N) * ones(N,1);
% scale(1) = sqrt(1/N);
% n = linspace(0,N-1,N)';
% F2 = (ones(N,1) * scale') .* cos((n + 0.5) * n' * pi / N );
% sum(sum(abs(F2 - F') < 0.000000001)) % Verified

% y = randn(N + 1, 1);
% Y_dct1 = dct_type1(y, N+1);
% Y_idct1 = idct_type1(y, N+1);
%  
% H = F' * diag(y(1:end-1)) * F;
% [V, D] = eig(H);
% 
% Hhat = (toeplitz(Y_idct1(1:end-1)) + hankel(Y_idct1(2:end), flipud(Y_idct1(2:end)))) / 2;
% [Vhat, Dhat] = eig(Hhat);
% 
% sum(sum(abs(Hhat - H) < 0.000000000001))

% This is how the fcn in python would get it
x = randn(N, 1);
z = [x; 0];
Z_dct1 = dct_type1(z, N+1);
Z_idct1 = idct_type1(z, N+1);
 
H = F' * diag(x) * F;
[V, D] = eig(H);

Hhat = (toeplitz(Z_idct1(1:end-1)) + hankel(Z_idct1(2:end), flipud(Z_idct1(2:end)))) / 2;
[Vhat, Dhat] = eig(Hhat);

sum(sum(abs(Hhat - H) < 0.000000000001))



