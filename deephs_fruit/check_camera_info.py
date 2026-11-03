import spectral
import os
from spectral.io import envi

# 读取VIS相机的一个.hdr文件（需要同时指定.bin文件）
hdr_path = 'data/Avocado/VIS/day_01/avocado_day_01_01_front.hdr'
bin_path = 'data/Avocado/VIS/day_01/avocado_day_01_01_front.bin'

img = envi.open(hdr_path, image=bin_path)

print('=== VIS Camera Info ===')
print(f'波段数 (nbands): {img.nbands}')
print(f'图像尺寸: {img.nrows} x {img.ncols}')

# 读取波长信息
metadata = img.metadata
if 'wavelength' in metadata:
    wavelengths = [float(w) for w in metadata['wavelength']]
    print(f'波长范围: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm')
    print(f'前5个波段波长: {[f"{w:.1f}" for w in wavelengths[:5]]}')
    print(f'后5个波段波长: {[f"{w:.1f}" for w in wavelengths[-5:]]}')
else:
    print('波长信息不可用')

print()
print('=== 对比 Specim FX 10 ===')
print('期望: 224 bands, 400-1000 nm')
print(f'实际: {img.nbands} bands')
if 'wavelength' in metadata:
    print(f'实际波长范围: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm')
