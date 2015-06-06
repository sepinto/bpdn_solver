clear all
n = 1024; F = dftmtx(n); F_unitary = F / sqrt(n);

f = rand(n, 1); % Our signal
h = rand(n, 1); % The thing we're convolving it with

lambda = F * h; % For us, this is 1 / sqrt(masked_energy) -- in freq domain
H = conj(F_unitary) * diag(lambda) * F_unitary;

vec1 = cconv(h,f,n);
vec2 = H * f;
err = norm(vec2 - vec1);

sprintf('||Hf - f*h||_2 = %f', err)
