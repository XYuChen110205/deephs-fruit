import spectral

# 读取VIS相机的一个.hdr文件
hdr_path = r'E:\DeepHS-Fruit\deephs_fruit\data\Avocado\VIS\day_01\avocado_day_01_01_front.hdr'
img = spectral.open_image(hdr_path)

print(f'=== VIS Camera Info ===')
print(f'波段数 (nbands): {img.nbands}')
print(f'波长范围: {img.bands.centers[0]:.1f} - {img.bands.centers[-1]:.1f} nm')
print(f'图像尺寸: {img.nrows} x {img.ncols}')
print()
print(f'前10个波段波长: {[f"{w:.1f}" for w in img.bands.centers[:10]]}')
print(f'后10个波段波长: {[f"{w:.1f}" for w in img.bands.centers[-10:]]}')
