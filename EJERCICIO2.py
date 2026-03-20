import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_gray(path, size=None):
 
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def binarize(img):
   
    _, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return bw.flatten().astype(np.uint8)

def resize_secret_to_capacity(secret_img, capacity_bits):
   
    h_s, w_s = secret_img.shape
    num_pixels = h_s * w_s
    if num_pixels <= capacity_bits:
        return secret_img

    lado = int(np.floor(np.sqrt(capacity_bits)))
    if lado < 1:
        raise RuntimeError("Carrier demasiado pequeño para ocultar ni un solo bit.")
    return cv2.resize(secret_img, (lado, lado), interpolation=cv2.INTER_AREA)

def embed_sign_flip_minmag(carrier_img, secret_img):
    """
    Embedding “cambio de signo” en coeficientes de magnitud mínima:
    1) Binariza la secreta (0/1 por píxel).
    2) Calcula FFT2D + FFTSHIFT de la portadora.
    3) Construye lista de coeficientes (mag, flat_idx) para todos los (i,j) ≠ DC.
    4) Ordena la lista por mag ascendente (menor a mayor).
    5) capacity_bits = len(coef_list)//2. Si la secreta tiene más pixeles que capacity_bits,
       la redimensiona (manteniendo proporción cuadrada) para que “cabe” sin error.
    6) Toma los primeros num_coef_needed = (#pixeles de la secreta redimensionada) coeficientes,
       e invierte el signo de real e imag para codificar cada bit.
       Conjuga el coeficiente espejo para mantener la hermiticidad.
    7) Reconstruye la imagen estego con IFFT.
    Devuelve:
      - stego_float: imagen estego en float32 ([0..255]).
      - indices_used: lista de (i,j) donde se incrustó cada bit (orden interno = orden de los bits).
      - secret_shape: (h_s, w_s) finales de la imagen secreta usada.
    """
    h, w = carrier_img.shape
    ch, cw = h // 2, w // 2

    # 1) Binarizar la imagen secreta
    bits_initial = binarize(secret_img)
    bits_secret = bits_initial.size

    # 2) FFT2D + FFTSHIFT de la portadora
    fft = np.fft.fft2(carrier_img.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    real = np.real(fft_shift).astype(np.float32)
    imag = np.imag(fft_shift).astype(np.float32)

    # 3) Crear lista de todos los coeficientes (i,j) ≠ DC con (mag_abs, flat_idx)
    mag = np.abs(fft_shift).flatten()
    idx_flat = np.arange(h * w)
    dc_flat = ch * w + cw
    # Marcar DC con mag = +∞ para que quede al final tras ordenar
    mag[dc_flat] = np.inf

    coef_list = [(mag_val, flat) for mag_val, flat in zip(mag, idx_flat) if flat != dc_flat]

    # 4) Ordenar por mag ascendente (de menor a mayor)
    coef_list.sort(key=lambda x: x[0])

    # 5) calcular capacidad: cada bit ocupa un coef y su simétrico
    total_coef = len(coef_list)
    capacity_bits = total_coef // 2

    # Si la secreta es demasiado grande, redimensionar
    if bits_secret > capacity_bits:
        secret_resized = resize_secret_to_capacity(secret_img, capacity_bits)
        bits_initial = binarize(secret_resized)
        bits_secret = bits_initial.size
    else:
        secret_resized = secret_img

    num_coef_needed = bits_secret

    # 6) Seleccionar los primeros num_coef_needed coeficientes (menor mag)
    if num_coef_needed > len(coef_list):
        # (Esto no debería ocurrir, porque redimensionamos arriba)
        raise RuntimeError("Carrier demasiado pequeño para ocultar todos los bits de la secreta.")

    # Extraer los índices “flat” (planos) y convertir a coordenadas (i,j)
    selected = [flat for (_, flat) in coef_list[:num_coef_needed]]
    indices_used = [(flat // w, flat % w) for flat in selected]

    # Insertar cada bit invirtiendo signo de (real, imag)
    for k in range(num_coef_needed):
        bit = int(bits_initial[k])
        i, j = indices_used[k]
        a_orig = real[i, j]
        b_orig = imag[i, j]
        signo = -1.0 if (bit == 1) else +1.0
        a_new = signo * abs(a_orig)
        b_new = signo * abs(b_orig)
        real[i, j] = a_new
        imag[i, j] = b_new

        # Simetría conjugada en (sym_i, sym_j)
        u, v = i - ch, j - cw
        sym_i = (ch - u) % h
        sym_j = (cw - v) % w
        real[sym_i, sym_j] = a_new
        imag[sym_i, sym_j] = -b_new

    # 7) Reconstruir la estego con IFFT
    new_fft_shift = real + 1j * imag
    new_fft = np.fft.ifftshift(new_fft_shift)
    stego_complex = np.fft.ifft2(new_fft)
    stego_float = np.real(stego_complex).astype(np.float32)

    return stego_float, indices_used, secret_resized.shape

def extract_sign_flip_minmag(stego_float, secret_shape, indices_used):
    """
    Decodificador:
    - Calcula la TF2D centrada de la estego.
    - Para cada (i,j) en indices_used recupera bit=0 si Re{F(i,j)} ≥ 0, o 1 si < 0.
    - Reconstruye la imagen secreta (0/255) con forma secret_shape.
    """
    h, w = stego_float.shape
    fft_rec = np.fft.fft2(stego_float.astype(np.float32))
    fft_shift = np.fft.fftshift(fft_rec)
    real_rec = np.real(fft_shift).astype(np.float32)

    total_bits = secret_shape[0] * secret_shape[1]
    bits_rec = np.zeros(total_bits, dtype=np.uint8)
    for k in range(total_bits):
        i, j = indices_used[k]
        a_r = real_rec[i, j]
        bits_rec[k] = 0 if a_r >= 0 else 1

    recovered = (bits_rec.reshape(secret_shape) * 255).astype(np.uint8)
    return recovered

if __name__ == "__main__":
    # 1) Cargar imágenes
    carrier = load_gray("original.png")  # Ej: 512×512
    secret  = load_gray("secreta.png")   # Ej: 256×256 o mayor (se ajustará automáticamente)

    # 2) Incrustar usando coeficientes de magnitud mínima
    stego_float, idxs, sec_shape = embed_sign_flip_minmag(carrier, secret)
    stego_uint8 = np.clip(stego_float, 0, 255).astype(np.uint8)
    cv2.imwrite("estego_minmag_final.png", stego_uint8)

    # 3) Extraer la imagen secreta
    recovered = extract_sign_flip_minmag(stego_float, sec_shape, idxs)
    cv2.imwrite("secreta_recuperada_minmag_final.png", recovered)

    # 4) Ajustar secreta para mostrarla (utilizamos sec_shape)
    secret_adjusted = cv2.resize(secret, (sec_shape[1], sec_shape[0]), interpolation=cv2.INTER_AREA)

    # 5) Métricas
    _, bw = cv2.threshold(secret_adjusted, 127, 1, cv2.THRESH_BINARY)
    bits_orig = bw.flatten().astype(np.uint8)
    bits_rec  = (recovered.flatten() > 127).astype(np.uint8)
    porcentaje = np.sum(bits_orig == bits_rec) / bits_orig.size * 100
    print(f"Porcentaje de bits recuperados: {porcentaje:.2f}%")

    valor_psnr = psnr(carrier, stego_uint8)
    print(f"PSNR (carrier vs estego): {valor_psnr:.2f} dB")

    # 6) Mostrar resultados
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Portadora Original")
    plt.imshow(carrier, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Secreta Ajustada")
    plt.imshow(secret_adjusted, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Estego (mag mínima)")
    plt.imshow(stego_uint8, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Secreta Recuperada")
    plt.imshow(recovered, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()