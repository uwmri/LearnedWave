"""
Gradient calibration using multi-slice thin slice method

Translated from PCVIPR::calc_trajectory

"""

import time
import numpy as np
import math
import h5py
import os
import matplotlib.pyplot as plt
import cupy as cp
from bitrev import get_bitrev_order


# scan parameters
echo_frac = 0.75
radius = 6
helix_cycles = 8
fov = 220
gamma = 4257.59    # Hz/G
xres = 256
scanner = 'wimr2'
scan_root = r'I:\Data\phantomacrctang_00740_2023-08-08'
scan_dir = fr'I:\Data\phantomacrctang_00740_2023-08-08\00740_00003_515_r6c8\raw_data'
opt_dir = r'I:\code\2Dsampling'

PC = False
num_slices = 64    # 64 slices each for G+ and G-
num_axis = 3
off_dist = 8.0  # rhuser05 [mm], here in cm
cal_ampmod = 1  # 4/25 scan"debug4" onwards, removed readout_scale when calibrating y and z gradients
area_to_phase = 2* math.pi * gamma * 1e-6
area_to_kspace = 1/ cal_ampmod * fov/10. *gamma* 1e-6

# set b0_eddy_center to 1 during scan, so oc_shifts are all 0.
# oc_shift = np.array([-25.62 , 0., 0.])
oc_shift = np.array([0. , 0., 0.]) # rhuser24, 25, 26 are x, y, z shift. All 0 if b0_eddy_center =1
oc_shift /= 10.

# For getting the expected trajectory. Can also use host side calc, which is a bit less close.
RwaveRSP = f'Rwave_{rhkacq_uid}.txt'    # this file comes from @RSP
gy, gz = np.loadtxt(os.path.join(scan_root, RwaveRSP), delimiter=",", unpack=True)

# 07/10/2023, fixed oprbw calc, need to double sample the Rwave from RSP
dt = 4e-3
dt2 = 2e-3
t = np.arange(0, gy.shape[0])*dt
t_interp = np.arange(0, gy.shape[0]*2)*dt2
gy = np.interp(t_interp, t, gy)
gz = np.interp(t_interp, t, gz)

# instruction amplitude to G/cm
scale_instr = 1/ 4755.587891  # scale_instr = max_pg_wamp/helix_target
gy *= scale_instr
gz *= scale_instr

pw_gxw = gy.shape[0] * dt2
a_gxw = 1e7 * echo_frac / (gamma * (fov/xres)) / pw_gxw     # 0.59 for r1c20

if PC:
    num_pulses = 3      # gx1, gxc, gxw
else:
    num_pulses = 1
num_calib = 2 * num_slices * num_pulses * num_axis

calibfile = h5py.File(name=os.path.join(scan_dir, 'calib.h5'), mode='r')
calib = calibfile['calib_r'][()] + 1j* calibfile['calib_i'][()]
num_coils = calib.shape[0]
nro = calib.shape[1]
calib0 = calib.reshape((num_coils, nro, num_axis,-1))
if PC:
    calib = calib.reshape((num_coils, nro, num_axis, num_pulses, num_slices, 2))
    calib = calib[:,:,:, -1, :, :]
else:
    calib = calib.reshape((num_coils, nro, num_axis, num_slices, 2))

# coords from waveform
wave_ky0 = radius * np.sin(-math.pi * helix_cycles * (echo_frac-0.5)/0.5)
wave_kz0 = radius * np.cos(-math.pi * helix_cycles * (echo_frac-0.5)/0.5)


kxmax = 1e-7 * fov * gamma * pw_gxw * a_gxw
kx = np.linspace(-kxmax * (echo_frac-.5)/echo_frac, kxmax * (1 -  (echo_frac-.5)/echo_frac), gy.shape[0])  # for 0.75 echo
ky = 1e-4 * fov * gamma * (np.add.accumulate(gy) * dt2) + wave_ky0
kz = 1e-4 * fov * gamma * (np.add.accumulate(gz) * dt2) + wave_kz0
helix = np.stack((kx, ky, kz), axis=1)
helixplt = plt.figure()
ax = helixplt.add_subplot(projection='3d')
ax.scatter(helix[:,0], helix[:,1], helix[:,2])
ax.set_box_aspect(aspect = (1,1,1))
plt.show()



# delta S for gxw, gyHelix, gzHelix
# the slices were bitrev ordered when acquired by default
S_diff = calib[:,:,:,:,0] * np.conj(calib[:,:,:,:,1])

order = get_bitrev_order(num_slices)

S_diff = S_diff.sum(axis=0)
S_diff_seq = np.zeros_like(S_diff)

for i in range(num_slices):
    idx = order[i]
    S_diff_seq[:,:,idx] = S_diff[:,:,i]

plt.figure();plt.plot(np.angle(S_diff_seq[10,1,:]))
plt.title('before subtract expected')
plt.show()



############################################## Grid search #############################################################
num_steps = 6


expected_area = helix / area_to_kspace
temp = np.tile(np.expand_dims(expected_area, -1), num_slices)
temp = cp.asarray(temp)


S_diff_seq = cp.asarray(S_diff_seq)
Grad_G = cp.zeros((nro, num_axis), dtype=cp.float32)
Grad_B = cp.zeros((nro, num_axis), dtype=cp.float32)
loss = cp.zeros((num_axis, nro, num_steps, num_slices), dtype=cp.complex64)
for dir in range(1,num_axis):
    x = cp.linspace(-off_dist, off_dist, num_slices)
    x += oc_shift[dir]
    S_diff_dir = S_diff_seq[:, dir, :] / cp.exp(1j * (2 * temp[:, dir, :] * x * area_to_phase))

    t0 = time.time()
    for i in range(nro):
        act_b0 = 0.
        act_area = 0.
        min_cost = cp.inf
        min_cost_sl = 0. + 0.j
        for step in range(num_steps):
            scale = 1/(8**step)
            b0_min = -math.pi /4 * scale + act_b0
            b0_max = math.pi /4 * scale + act_b0

            min_area = -15 * scale + act_area
            max_area = 15 * scale + act_area

            for b0 in cp.linspace(b0_min, b0_max, num=32):
                for area in cp.linspace(min_area, max_area, num=32):
                    cost_sl = S_diff_dir[i] - cp.abs(S_diff_dir[i]) * cp.exp(1j*(2*b0 + 2*area*x*area_to_phase))
                    cost = cp.sum(cp.abs(cost_sl), axis=0)

                    if (cost < min_cost):
                        min_cost = cost
                        min_cost_sl = cost_sl
                        act_area = area
                        act_b0 = b0
                        # print(f'min_cost {min_cost}, b0{act_b0}, area {act_area}')
            loss[dir, i, step] = min_cost_sl
            Grad_G[i, dir] = act_area + expected_area[i, dir]
            Grad_B[i, dir] = act_b0
        # plt.figure();plt.plot(np.asarray(loss));plt.show()
    t = time.time()
    print(f'{nro} takes {t-t0}s')

Grad_G = Grad_G.get()
Grad_B = Grad_B.get()

Grad_G[:,0] = expected_area[:, 0]
k = Grad_G * area_to_kspace



helixplt = plt.figure()
ax = helixplt.add_subplot(projection='3d')
ax.scatter(k[:, 0], k[:, 1], k[:, 2], c='r', marker='o')
ax.scatter(helix[:,0], helix[:,1], helix[:,2], c='b', marker='^')
ax.set_box_aspect(aspect=(1,1,1))
plt.show()

helixfile =  os.path.join(scan_root,f'helix_calibrated_r{radius}c{helix_cycles}echo{echo_frac}_fov{fov}.txt')
try:
    os.remove(helixfile)
except OSError:
    pass
np.savetxt(helixfile, k, fmt='%f', delimiter=",")

b0file =  os.path.join(scan_root,f'b0_calibrated_r{radius}c{helix_cycles}echo{echo_frac}_fov{fov}.txt')
try:
    os.remove(b0file)
except OSError:
    pass
np.savetxt(b0file,  Grad_B, fmt='%f', delimiter=",")

sl = 21
fit = S_diff_dir[sl] - loss[1, sl, 1]
plt.figure();plt.plot(cp.abs(S_diff_dir[sl]).get());plt.plot( cp.abs(fit).get());plt.show()
plt.figure();plt.plot(cp.angle(S_diff_dir[sl]).get());plt.plot( cp.angle(fit).get());plt.show()

