# ── Importaciones ────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import urllib.request
import os, io, warnings

from scipy.stats import gamma
from skimage import data, exposure, filters, morphology, util
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte
from scipy import ndimage


def load_image(image_name:str, display:bool = False) -> Image:
    img = Image.open("img/"+image_name)
    if display:
        display_image(img)
    return img

def display_image(image:Image) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])

    plt.margins(0)
    plt.show()

def display_histogram_rgb(image:Image):
    r, g, b = image.split()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.hist(np.array(r).flatten(), bins=256, color='red', alpha=0.5)
    plt.title('Histograma canal rojo')
    plt.subplot(1, 3, 2)
    plt.hist(np.array(g).flatten(), bins=256, color='green', alpha=0.5)
    plt.title('Histograma canal verde')
    plt.subplot(1, 3, 3)
    plt.hist(np.array(b).flatten(), bins=256, color='blue', alpha=0.5)
    plt.title('Histograma canal azul')
    plt.tight_layout()
    plt.show()


def get_as_gray_scale(image:Image, display:bool) -> Image:
    image_gray =image.convert('L')
    if display:
        plt.imshow(image_gray, cmap='gray')
        plt.axis('off')
        plt.show()
    return  image_gray

def operator_sum(image:Image, constant:int) -> Image:
    image_sum = Image.fromarray(np.clip(np.array(image) + constant, 0, 255).astype('uint8'))
    return image_sum

def get_negative(image:Image) -> Image:
    image_negative = Image.fromarray(np.clip(255-np.array(image), 0, 255).astype('uint8'))
    return image_negative

def get_logarithmic_adjust(image:Image, constant:int) -> Image:
    c_log = (256 - 1) / np.log(1 + (256 - 1))
    image_logarithmic = Image.fromarray(np.clip(c_log * np.log(np.array(image) + 10), 0, 255).astype('uint8'))
    return image_logarithmic

def get_power_law(image:Image, gamma:float, scale:float) -> Image:
    image_gamma = Image.fromarray(np.clip(scale * np.power(np.array(image), gamma), 0, 255).astype('uint8'))
    return  image_gamma

def get_contrast_equilizing(image:Image, constant:int) -> Image:
    image_gray = image.convert('L')
    image_as_array = np.array(image_gray).astype('uint8')
    equalized_array = cv2.equalizeHist(image_as_array)
    return Image.fromarray(equalized_array)
    # f_piecewise = np.where(r < 50, r * 0.3,
    #                        np.where(r < 200, (r - 50) * (205 / 150) + 15,
    #                                 (r - 200) * 0.3 + 220))

#(r/(L-1))**2.5 * (L-1)
def get_contrast_equilizing_rgb(image: Image) -> Image:
    # Convertir PIL Image a array
    image_array = np.array(image)

    # Separar los 3 canales
    r, g, b = cv2.split(image_array)

    # Ecualizar cada canal
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # Recombinar los canales
    image_equalized = cv2.merge([r_eq, g_eq, b_eq])

    # Convertir de vuelta a PIL Image
    return Image.fromarray(image_equalized)

def funcion_a_trozos(image:Image):
    image_array = np.array(image)
    f_piecewise = np.where(image_array < 50, image_array +20,
                           np.where(image_array < 230, image_array,
                                    image_array-50))
    return Image.fromarray(np.clip(f_piecewise, 0, 255).astype('uint8'))

def compare_histograms(original:Image, enhanced:Image) -> None:
    original_array = np.array(original)
    enhanced_array = np.array(enhanced)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(original_array.flatten(), bins=256, color='gray', alpha=0.5)
    plt.title('Histogram Original')
    plt.subplot(1, 2, 2)
    plt.hist(enhanced_array.flatten(), bins=256, color='gray', alpha=0.5)
    plt.title('Histograma actual')
    plt.tight_layout()
    plt.show()

def enhacement_curve(original:Image, enhanced:Image) -> None:
    # Convertir PIL Image a array
    image_array_original = np.array(original)
    image_array_enhanced = np.array(enhanced)

    # Separar canales
    r_o, g_o, b_o = cv2.split(image_array_original)
    r_e, g_e, b_e = cv2.split(image_array_enhanced)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(
        'Relación de intensidades original vs transformada',
        fontsize=13
    )

    ax.scatter(r_o.ravel(), r_e.ravel(), color='red', alpha=0.15, s=3, label='Canal R')
    ax.scatter(g_o.ravel(), g_e.ravel(), color='green', alpha=0.15, s=3, label='Canal G')
    ax.scatter(b_o.ravel(), b_e.ravel(), color='blue', alpha=0.15, s=3, label='Canal B')

    ax.set_xlabel('Intensidad de entrada f(x,y)', fontsize=11)
    ax.set_ylabel('Intensidad de salida g(x,y)', fontsize=11)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True)
    ax.set_aspect('equal')

    plt.show()

def calculate_contrast_indicators(image: Image) -> dict:
    """
    Calcula indicadores globales de contraste para una imagen.

    La imagen se convierte a escala de grises para calcular métricas globales.
    """
    image_gray = image.convert('L')
    image_array = np.array(image_gray).astype(np.float64)

    i_min = np.min(image_array)
    i_max = np.max(image_array)
    mean = np.mean(image_array)
    std = np.std(image_array)
    variance = np.var(image_array)

    absolute_contrast = i_max - i_min

    michelson_contrast = (
        (i_max - i_min) / (i_max + i_min)
        if (i_max + i_min) != 0
        else 0
    )

    rms_contrast = std

    normalized_rms_contrast = (
        std / mean
        if mean != 0
        else 0
    )

    dynamic_range = i_max - i_min

    histogram, _ = np.histogram(image_array.flatten(), bins=256, range=(0, 255), density=True)
    histogram = histogram[histogram > 0]
    entropy = -np.sum(histogram * np.log2(histogram))

    return {
        "Intensidad mínima": i_min,
        "Intensidad máxima": i_max,
        "Media": mean,
        "Desviación estándar": std,
        "Varianza": variance,
        "Contraste absoluto": absolute_contrast,
        "Contraste Michelson": michelson_contrast,
        "Contraste RMS": rms_contrast,
        "Contraste RMS normalizado": normalized_rms_contrast,
        "Rango dinámico": dynamic_range,
        "Entropía": entropy
    }


def compare_contrast_indicators(original: Image, enhanced: Image) -> None:
    """
    Compara indicadores de contraste entre una imagen original y una imagen transformada.
    """
    original_indicators = calculate_contrast_indicators(original)
    enhanced_indicators = calculate_contrast_indicators(enhanced)

    table_data = []

    for indicator in original_indicators:
        original_value = original_indicators[indicator]
        enhanced_value = enhanced_indicators[indicator]
        difference = enhanced_value - original_value

        percentage_change = (
            (difference / original_value) * 100
            if original_value != 0
            else 0
        )

        table_data.append([
            indicator,
            f"{original_value:.4f}",
            f"{enhanced_value:.4f}",
            f"{difference:.4f}",
            f"{percentage_change:.2f}%"
        ])

    headers = ["Indicador", "Valor Original", "Valor Mejorado", "Diferencia", "Cambio (%)"]

    rows = [headers] + table_data

    column_widths = [
        max(len(str(row[column_index])) for row in rows)
        for column_index in range(len(headers))
    ]

    separator = "+" + "+".join("-" * (width + 2) for width in column_widths) + "+"

    def format_row(row):
        return "|" + "|".join(
            f" {str(value):<{column_widths[index]}} "
            for index, value in enumerate(row)
        ) + "|"

    print(separator)
    print(format_row(headers))
    print(separator)

    for row in table_data:
        print(format_row(row))

    print(separator)






image_10=load_image("1002.png", display=False)
#display_histogram_rgb(image_10)
#get_as_gray_scale(image_10, True)
#display_image(operator_sum(image_10, 100))
#display_image(get_negative(image_10))
#display_image(get_logarithmic_adjust(image_10, 1))



# gamma =0.4
# scale = (256-1)/pow((256-1), gamma)
# display_image(get_power_law(image_10, gamma, scale))


#nhanced_image = get_contrast_equilizing_rgb(image_10)
#enhanced_image= get_logarithmic_adjust(image_10, 10)
gamma = 0.4
scale = (256-1)/pow((256-1), gamma)
enhanced_image=get_power_law(image_10, gamma, scale)
#enhanced_image = funcion_a_trozos(image_10)
display_image(image_10)
display_image(enhanced_image)
# compare_histograms(image_10, enhanced_image)
display_histogram_rgb(image_10)
display_histogram_rgb(enhanced_image)
enhacement_curve(image_10, enhanced_image)
compare_contrast_indicators(image_10, enhanced_image)


