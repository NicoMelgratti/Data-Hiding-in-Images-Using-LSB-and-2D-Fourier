import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_gray(path, size=None):
    """
    Carga una imagen en escala de grises y opcionalmente la redimensiona.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def binarize(img):
    """
    Convierte una imagen (0..255) en bits {0,1} usando umbral 127.
    Devuelve un array de bits en orden “plano”.
    """
    _, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return bw.flatten().astype(np.uint8)

def embed_parity(a_val, bit, delta):
    """
    A partir de un valor real (a_val) y un bit {0,1}, 
    calcula q = round(|a_val|/delta), ajusta paridad q%2 = bit, 
    y reconstruye signo(a_val) * (q*delta).
    """
    signo = 1.0 if a_val >= 0 else -1.0
    q = int(round(abs(a_val) / delta))
    if (q % 2) != bit:
        q += 1
    return signo * (q * delta)

def adjust_secret_size(carrier_shape, secret_img):
    
    """
    Si la cantidad de píxeles de secret_img supera la capacidad, redimensiona a un cuadrado máximo.
    Capacidadbits = floor(((h*w) - 1) / 2) 
    """
    h, w = carrier_shape
    capacidad_bits = (h * w - 1) // 2
    h_s, w_s = secret_img.shape
    num_pixeles = h_s * w_s

    if num_pixeles <= capacidad_bits:
        return secret_img

    max_pixeles = capacidad_bits
    lado = int(np.floor(np.sqrt(max_pixeles)))
    if lado < 1:
        raise RuntimeError("Imposible ocultar ni un solo bit en la portadora.")
    # Redimensionar la imagen secreta a (lado × lado)
    secret_resized = cv2.resize(secret_img, (lado, lado), interpolation=cv2.INTER_AREA)
    return secret_resized

def stego_fixed_delta(carrier_img, secret_img, delta):
    """
    Inserta cada bit de secret_img (binarizada) en un único coeficiente (y su espejo conjugado).
    Si la imagen secreta es demasiado grande, la redimensiona automáticamente.
    """
    # 1) Ajustar el tamaño de la secreta si hace falta
    secret_adjusted = adjust_secret_size(carrier_img.shape, secret_img)
    # 2) Binarizamos la imagen secreta (después de redimensionar, si aplicó)
    _, bw = cv2.threshold(secret_adjusted, 127, 1, cv2.THRESH_BINARY)
    bits = bw.flatten().astype(np.uint8)
    total_bits = bits.size


    # 3) FFT2D + FFTSHIFT de la portadora
    fft = np.fft.fft2(carrier_img.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    real = np.real(fft_shift).astype(np.float32)
    imag = np.imag(fft_shift).astype(np.float32)


    h, w = carrier_img.shape
    ch, cw = h // 2, w // 2

    # 4) Ordenar índices por magnitud compleja (excluyendo DC)
    mag = np.abs(fft_shift).flatten()
    idx_flat = np.arange(h * w)
    dc_flat = ch * w + cw
    mag[dc_flat] = -1  # nunca elegimos DC

    sorted_idx = idx_flat[np.argsort(-mag)]

    # 5) Seleccionar exactamente total_bits posiciones (sin redundancia)
    selected = []
    used = set()
    
    for flat in sorted_idx:
        if len(selected) >= total_bits:
            break
        if flat == dc_flat:
            continue

        i = flat // w
        j = flat % w
        
        # coordenadas relativas (u,v)
        
        u = i - ch
        v = j - cw
        sym_i = (ch - u) % h
        sym_j = (cw - v) % w
        sym_flat = sym_i * w + sym_j

        if (flat in used) or (sym_flat in used):
            continue

        selected.append(flat)
        used.add(flat)
        used.add(sym_flat)

    if len(selected) < total_bits:
        raise RuntimeError(
            f"No hay suficientes coeficientes para ocultar {total_bits} bits; "
            f"solo disponemos de {len(selected)} posiciones."
        )

    # Convertir a lista de tuplas (i,j)
    indices_emb = [(flat // w, flat % w) for flat in selected[:total_bits]]

    # 6) Insertar cada bit en un coeficiente (y mantener su simétrico conjugado)
    for k in range(total_bits):
        bit = int(bits[k])
        i, j = indices_emb[k]

        a_orig = real[i, j]
        b_orig = imag[i, j]

        a_new = embed_parity(a_orig, bit, delta)
        b_new = embed_parity(b_orig, bit, delta)

        real[i, j] = a_new
        imag[i, j] = b_new

        # Mantener hermiticidad en la posición simétrica
        u = i - ch
        v = j - cw
        sym_i = (ch - u) % h
        sym_j = (cw - v) % w

        real[sym_i, sym_j] = a_new
        imag[sym_i, sym_j] = -b_new

    # 7) Reconstruir con IFFT
    new_fft_shift = real + 1j * imag
    new_fft = np.fft.ifftshift(new_fft_shift)
    stego_complex = np.fft.ifft2(new_fft)
    stego_float = np.real(stego_complex).astype(np.float32)

    return stego_float, indices_emb

def extract_fixed_delta(stego_float, delta, secret_shape, indices_emb):
    """
    Dada la imagen estego (float32), extrae cada bit leyendo la paridad
    de round(|Re{F(i,j)}|/delta). Redundancia = 1.
    Retorna la imagen secreta reconstruida (0/255) con la forma `secret_shape`.
    """
    h, w = stego_float.shape

    fft = np.fft.fft2(stego_float.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    real_rec = np.real(fft_shift).astype(np.float32)

    total_bits = secret_shape[0] * secret_shape[1]
    bits_rec = np.zeros(total_bits, dtype=np.uint8)

    for k in range(total_bits):
        i, j = indices_emb[k]
        a_r = real_rec[i, j]
        q = int(round(abs(a_r) / delta))
        bits_rec[k] = q % 2

    recovered = (bits_rec.reshape(secret_shape) * 255).astype(np.uint8)
    return recovered

# ----------------------- EJECUCIÓN PRINCIPAL -----------------------

# ...código anterior...

if __name__ == "__main__":
    # Parámetros fijos
    delta_value = 20

    # 1) Cargar imágenes (escala de grises)
    carrier = load_gray("original.png")       # ej. 512×512 o similar
    secret  = load_gray("secreta.png")         # ej. puede ser mayor, se redimensionará si hace falta

    # 2) Incrustar (Redundancia = 1, δ = 20)
    stego_float, idxs = stego_fixed_delta(carrier, secret, delta_value)

    # Convertir stego_float a uint8 para guardar/mostrar
    stego_uint8 = np.clip(stego_float, 0, 255).astype(np.uint8)
    cv2.imwrite("estego_fijo.png", stego_uint8)

    # 3) Extraer
    secret_adjusted = adjust_secret_size(carrier.shape, secret)
    recovered = extract_fixed_delta(stego_float, delta_value, secret_adjusted.shape, idxs)
    cv2.imwrite("secreta_recuperada_fija.png", recovered)

    # 4) Métricas: PSNR y exactitud de bits
    _, secret_bw = cv2.threshold(secret_adjusted, 127, 1, cv2.THRESH_BINARY)
    bits_orig = secret_bw.flatten().astype(np.uint8)
    bits_rec = (recovered.flatten() > 127).astype(np.uint8)

    # Calcular porcentaje de recuperación
    iguales = np.sum(bits_orig == bits_rec)
    total = bits_orig.size
    porcentaje = iguales / total * 100
    print(f"Porcentaje de píxeles recuperados correctamente: {porcentaje:.2f}%")

    # PSNR
    psnr_val = psnr(carrier, stego_uint8)
    print(f"PSNR (carrier vs estego_fijo): {psnr_val:.2f} dB")

    # 5) Mostrar resultado
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Portadora Original")
    plt.imshow(carrier, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Estego (δ={delta_value})")
    plt.imshow(stego_uint8, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Secreto Recuperado")
    plt.imshow(recovered, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()