clear all
fs = 48000; n = 1024;

% Sinusoid
% f = sin(2 * pi * 480 * n / fs);
% F = fft(f)
% F_dct = dct(f)

% Forced sparsity in DFT domain
% half_F = [0 0 0 0 1 zeros(1,n/2-5)];
% F = [half_F 0 fliplr(half_F(2:end))];
% f = ifft(F);
% F_dct = dct(f);

% Forced sparsity in DCT domain
F_dct = [zeros(1,20)  1 zeros(1,n-21)];
f = idct(F_dct);
F = fft(f);

figure(1)
subplot(3,1,1)
plot(f)
subplot(3,1,2)
plot(abs(F))
subplot(3,1,3)
plot(F_dct)