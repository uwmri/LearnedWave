import numpy as np
import os

def get_bitrev_order(num_proj):
    # ordering
    num_bits = num_proj.bit_length()
    tmp_bitrev = np.zeros(num_proj, dtype='int32')
    bitrev = np.zeros(num_proj, dtype='int32')
    for j in range(num_proj):
        tmp = j
        for i in range(num_bits):
            tmp_bitrev[j] += pow(2, num_bits - i - 1) * (tmp % 2)
            tmp /= 2

    bitrev[0] = tmp_bitrev[0]
    curr_high = 0

    for j in range(1, num_proj):
        tmp = num_proj + 1
        diff_element = 0
        for i in range(1, num_proj):
            if tmp_bitrev[i] > curr_high:
                diff = tmp_bitrev[i] - curr_high
                if diff < tmp:
                    diff_element = i
                    tmp = diff
        curr_high = tmp_bitrev[diff_element]
        bitrev[diff_element] = j

    return bitrev


Ntrial = 5242
Nepoch = 20
wave_dir = rf'S:\code\LearnedWave\trained_models\{Ntrial}'
kxc, kyc = np.loadtxt(os.path.join(wave_dir,f'centers_opt_3D_4385_{Ntrial}_{Nepoch}.txt'), delimiter=',', unpack=True )
scale, rot = np.loadtxt(os.path.join(wave_dir,f'scale_rot_{Ntrial}_{Nepoch}.txt'), delimiter=',', unpack=True)
num_views = kxc.shape[0]
order = get_bitrev_order(num_views)

kxc_re = np.zeros_like(kxc)
kyc_re = np.zeros_like(kyc)
scale_re = np.zeros_like(scale)
rot_re = np.zeros_like(rot)
kw_re = np.zeros_like(kw)
for i in range(num_views):
    kxc_re[i] = kxc[order[i]]
    kyc_re[i] = kyc[order[i]]
    scale_re[i] = scale[order[i]]
    rot_re[i] = rot[order[i]]
    kw_re[i] = kw[order[i]]

centers = np.stack((kxc_re, kyc_re), axis=1)
scalerot = np.stack((scale_re, rot_re), axis=1)


centersfile = os.path.join(wave_dir, f'centers_opt_3D_4385_{Ntrial}_{Nepoch}_permute.txt')
try:
    os.remove(centersfile)
except OSError:
    pass
np.savetxt(centersfile, centers, fmt='%f', delimiter=",")

scalerotfile = os.path.join(wave_dir, f'scale_rot_{Ntrial}_{Nepoch}_permute.txt')
try:
    os.remove(scalerotfile)
except OSError:
    pass
np.savetxt(scalerotfile, scalerot, fmt='%f', delimiter=",")