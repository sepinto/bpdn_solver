function [ result ] = idct_type1( x, N )
% See http://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms
result = (2 / (N - 1)) * dct_type1(x, N);
end

