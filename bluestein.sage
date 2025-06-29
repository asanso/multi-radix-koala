def naive_fft(a, omega, inverse=False):
    n = len(a)
    w = omega
    if inverse:
        w = omega^(-1)
    # precompute powers w^k
    w_pow = [1]
    for k in range(1, n):
        w_pow.append(w_pow[-1] * w)
    # main DFT loop
    out = []
    for t in range(n):
        acc = 0
        for x, ax in enumerate(a):
            acc += ax * w_pow[(x * t) % n]
        out.append(acc)
    # scale for inverse
    if inverse:
        out = [y / n for y in out]
    return out

#-------------------------
# Bluestein's algorithm
#-------------------------
p = 2^64 - 2^32 + 1
F = GF(p)

# original FFT size and convolution size
s = 5
pow2 = 16
assert pow2 >= 2*s - 1

# roots of unity
omega = F(7)^((p - 1) / s)
omega_pow2 = F(7)^((p - 1) / pow2)
omega_inv = omega.inverse()

# input vector (in F)
f = [F.random_element() for i in range(s)]
assert len(f) == s

# compute chirps
inv2 = inverse_mod(2, s)
g = [omega^(j*j * inv2) for j in range(s)]
g_inv = [omega^(-j*j * inv2) for j in range(s)]
# pad convolution kernel with inverse chirp
ginv_pad = [F(0)] * pow2
ginv_pad[0] = g_inv[0]
for j in range(1, s):
    ginv_pad[j] = g_inv[j]
    ginv_pad[pow2 - j] = g_inv[j]
    
# scale input by forward chirp and pad
f_pad = [f[j] * g[j] for j in range(s)] + [F(0)] * (pow2 - s)

# FFT-based convolution
a_fft = naive_fft(f_pad, omega_pow2)
b_fft = naive_fft(ginv_pad, omega_pow2)
conv_fft = [a_fft[i] * b_fft[i] for i in range(pow2)]
conv = naive_fft(conv_fft, omega_pow2, inverse=True)

# recover final DFT values
X = [conv[j] * g[j] for j in range(s)]

# compare with naive DFT
dft_naive = naive_fft(f, omega)
print("Bluestein ->", [hex(v) for v in X])
print("Naive    ->", [hex(v) for v in dft_naive])
