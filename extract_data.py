import json

with open(r'e:\DeepHS-Fruit\HSI_Fruit_Maturity_Detection\HSI_Fruit_Maturity_Detection\templates\detection_records.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

record = data[0]
wavelengths = record['wavelengths']
spectral_curve = record['spectral_curve']

for w, s in zip(wavelengths, spectral_curve):
    print(f"{w} {s}")
