import numpy as np
from PIL import Image

def mensaje_a_bits(mensaje):
    # Agrega el símbolo de fin de mensaje
    mensaje += '&'
    # Convierte a lista de bits (string de 0s y 1s)
    bits = ''.join([format(ord(c), '08b') for c in mensaje])
    return bits

def ocultar_mensaje(imagen_path, mensaje, salida_path):
    # Cargar imagen en escala de grises
    img = Image.open(imagen_path).convert('L')
    datos = np.array(img)
    plano = datos.flatten()
    bits = mensaje_a_bits(mensaje)
    if len(bits) > len(plano):
        raise ValueError("El mensaje es demasiado largo para esta imagen.")
    # Modificar el LSB de cada píxel
    for i, bit in enumerate(bits):
        plano[i] = (plano[i] & 0xFE) | int(bit)
    # Reconstruir y guardar imagen estego
    datos_mod = plano.reshape(datos.shape)
    img_mod = Image.fromarray(datos_mod.astype(np.uint8))
    img_mod.save(salida_path)
    print(f"Imagen estego guardada en {salida_path}")
    

def extraer_mensaje(imagen_path):
    img = Image.open(imagen_path).convert('L')
    datos = np.array(img).flatten()
    bits = [str(pixel & 1) for pixel in datos]
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int(''.join(byte), 2))
        if char == '&':
            break
        chars.append(char)
    mensaje = ''.join(chars)
    return mensaje


import matplotlib.pyplot as plt

def mostrar_imagenes(original_path, estego_path):
    orig = Image.open(original_path).convert('L')
    estego = Image.open(estego_path).convert('L')
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(orig, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(estego, cmap='gray')
    axs[1].set_title('Estego')
    axs[1].axis('off')
    plt.show()
    
    # Codificar
ocultar_mensaje('original.png', 'hola como anda todo el mundo.', 'estego.png')

# Decodificar
mensaje = extraer_mensaje('estego.png')
print("Mensaje decodificado:", mensaje)

# Mostrar imágenes
mostrar_imagenes('original.png', 'estego.png')

