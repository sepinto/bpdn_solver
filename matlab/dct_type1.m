function [ result ] = dct_type1( x, N )
% See http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-I
% x needs to be column vector
k = linspace(0,N-1,N)';
n = linspace(1,N-2,N-2);
result = 0.5 * (x(1) + ((-1).^k) * x(end)) + cos(k * n * pi / (N - 1)) * x(2:end-1);
end

