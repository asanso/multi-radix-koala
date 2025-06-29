from sage.all import *
import time

# Field setup
p = 2**8 * 31 + 1
F = GF(p)

N1 = 16
N2 = 31
N = N1 * N2
omega = F.multiplicative_generator() ** ((p-1)//N)

# Radix-2 FFT for N1
def fft_radix2(a, omega):
    n = len(a)
    assert n & (n-1) == 0
    if n == 1:
        return a
    even = fft_radix2(a[0::2], omega**2)
    odd = fft_radix2(a[1::2], omega**2)
    result = [0]*n
    w = F(1)
    for i in range(n//2):
        result[i] = even[i] + w*odd[i]
        result[i+n//2] = even[i] - w*odd[i]
        w *= omega
    return result

# Naive DFT for N2
def naive_dft(a, omega):
    n = len(a)
    w_pows = [omega**i for i in range(n)]
    return [sum(a[j] * w_pows[(j*k)%n] for j in range(n)) for k in range(n)]

# Mixed-radix 1D Cooley-Tukey
def mixed_radix_fft(a, N1, N2, omega):
    temp = [F(0)]*len(a)
    # Step 1: N2 blocks of length N1
    for j2 in range(N2):
        block = [a[j2 + N2*j1] for j1 in range(N1)]
        block_fft = fft_radix2(block, omega**N2)
        for k1 in range(N1):
            temp[j2 + N2*k1] = block_fft[k1]
    # Step 2: Multiply by twiddle
    for k1 in range(N1):
        for j2 in range(N2):
            temp[j2 + N2*k1] *= omega**(j2*k1)
    # Step 3: N1 blocks of length N2
    output = [F(0)]*len(a)
    for k1 in range(N1):
        column = [temp[j2 + N2*k1] for j2 in range(N2)]
        column_fft = naive_dft(column, omega**N1)
        for k2 in range(N2):
            output[k1 + N1*k2] = column_fft[k2]
    return output

# Naive 1D DFT
def naive_1d_dft(a, omega):
    N = len(a)
    w_pows = [omega**i for i in range(N)]
    return [sum(a[j] * w_pows[(j*k)%N] for j in range(N)) for k in range(N)]

# Test the mixed-radix FFT
mat = [[F.random_element() for _ in range(N1)] for _ in range(N2)]
a_flat = [mat[i][j] for i in range(N2) for j in range(N1)]
# Polynomial in coeff form

# Polynomial in coeff form
R.<X> = PolynomialRing(F)
poly_from_flat = sum(a_flat[k]*X^k for k in range(N))
print(f"Polynomial:\n{poly_from_flat}")

start = time.time()
mixed_fft = mixed_radix_fft(a_flat, N1, N2, omega)
print(f"Mixed-radix FFT took {time.time()-start:.4f} seconds")

start = time.time()
naive_fft = naive_1d_dft(a_flat, omega)
print(f"Naive 1D DFT took {time.time()-start:.4f} seconds")

mismatch = any(mixed_fft[k] != naive_fft[k] for k in range(N))
if mismatch:
    print("❌ Mismatch between mixed-radix FFT and naive DFT!")
else:
    print("✅ Mixed-radix FFT matches naive DFT")
