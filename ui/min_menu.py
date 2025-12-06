import os

print("1. Detectar en FOTO")
print("2. Detectar en VIDEO")
opt = input("Elige opción: ")

if opt == "1":
    os.system("python detect_photo.py")
elif opt == "2":
    os.system("python detect_video.py")
else:
    print("Opción inválida")
