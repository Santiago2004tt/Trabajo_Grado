# -*- coding: utf-8 -*-
"""Interfaz gráfica mínima para estimar LPIPS antes de recodificar una imagen.

El archivo concentra en un único .py:
- extracción de características visuales;
- carga del preprocesamiento y del modelo entrenado;
- preparación de las variables de entrada;
- predicción anticipada de LPIPS;
- interfaz gráfica con Tkinter.

Los artefactos entrenados (.keras, .joblib y .json) deben permanecer como archivos
externos dentro de las carpetas configuradas; no se pueden sustituir por código.
"""

from pathlib import Path
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import circvar, entropy
from skimage import color, feature, filters, io
from skimage.util import img_as_float, img_as_ubyte
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
# Las rutas son relativas a la ubicación de este .py, no al directorio
# desde el que se ejecute el comando de Python.
BASE_DIR = Path(__file__).resolve().parent
DIR_ARTEFACTOS = BASE_DIR / "modelo_lpips_exportado"
DIR_RESULTADOS = BASE_DIR / "resultados_optimizacion"

RUTA_PREPROCESAMIENTO = DIR_ARTEFACTOS / "preprocesamiento.joblib"
RUTA_MODELO = DIR_RESULTADOS / "modelo_lpips_optimizado.keras"
RUTA_CONFIGURACION = DIR_RESULTADOS / "configuracion_modelo.json"

FORMATOS_VALIDOS = {"jpg", "png", "webp"}
CALIDADES_VALIDAS = {10, 30, 50, 70, 90}


# ============================================================
# VALIDACIÓN ESTRICTA DE LA IMAGEN ORIGINAL
# ============================================================
def inspeccionar_imagen(path_imagen):
    """Verifica formato real, extensión, modo RGB y dimensiones mínimas.

    Requisitos del prototipo:
    - El archivo debe ser JPEG/JPG, PNG o WebP.
    - La extensión debe corresponder al formato real del archivo.
    - El modo de color debe ser exactamente RGB (no L, P, RGBA, etc.).
    - Ancho y alto deben ser como mínimo de 64 píxeles.

    Devuelve los metadatos cuando la imagen cumple todos los requisitos;
    en caso contrario lanza ValueError con el motivo.
    """
    path_imagen = Path(path_imagen)

    if not path_imagen.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {path_imagen}")
    if not path_imagen.is_file():
        raise ValueError("La ruta seleccionada no corresponde a un archivo.")

    extension = path_imagen.suffix.lower().lstrip(".")
    if extension == "jpeg":
        extension = "jpg"
    if extension not in FORMATOS_VALIDOS:
        raise ValueError(
            "Formato no admitido. Selecciona una imagen JPG, PNG o WebP."
        )

    try:
        with Image.open(path_imagen) as imagen:
            formato_pillow = (imagen.format or "").upper()
            modo = imagen.mode
            ancho, alto = imagen.size
            imagen.verify()
    except Exception as exc:
        raise ValueError(f"No se pudo leer la imagen seleccionada: {exc}") from exc

    mapa_formatos = {"JPEG": "jpg", "JPG": "jpg", "PNG": "png", "WEBP": "webp"}
    formato_real = mapa_formatos.get(formato_pillow)
    if formato_real is None:
        raise ValueError(
            f"El contenido del archivo no es un formato compatible: {formato_pillow or 'desconocido'}."
        )

    errores = []
    if extension != formato_real:
        errores.append(
            f"La extensión del archivo es .{extension}, pero su formato real es {formato_real.upper()}. "
            "Cambia el nombre del archivo para que coincidan o selecciona otro."
        )
    if modo != "RGB":
        errores.append(
            f"La imagen no está en modo RGB (modo detectado: {modo}). "
            "Se requiere una imagen RGB de tres canales; no se admiten escala de grises, paleta ni RGBA."
        )
    if ancho < 64 or alto < 64:
        errores.append(
            f"La resolución es {ancho} × {alto} píxeles. El ancho y el alto deben ser, como mínimo, de 64 × 64 píxeles."
        )

    if errores:
        raise ValueError("\n\n".join(errores))

    return {
        "ruta": path_imagen,
        "formato": formato_real,
        "modo": modo,
        "ancho": ancho,
        "alto": alto,
    }

MODELO = None
PREPROCESAMIENTO = None
CONFIGURACION = None


# ============================================================
# FUNCIONES AUXILIARES DE LECTURA DE IMÁGENES
# ============================================================
def _leer_gris(path_imagen):
    """Lee una imagen como escala de grises normalizada."""
    imagen = io.imread(str(path_imagen), as_gray=True)
    return img_as_float(imagen)


def cargar_imagen_rgb(path_imagen):
    """Carga una imagen RGB normalizada; convierte gris y elimina alfa."""
    imagen = img_as_float(io.imread(str(path_imagen)))

    if imagen.ndim == 2:
        imagen = color.gray2rgb(imagen)
    elif imagen.ndim == 3 and imagen.shape[2] == 4:
        imagen = imagen[:, :, :3]
    elif imagen.ndim == 3 and imagen.shape[2] == 3:
        pass
    else:
        raise ValueError(
            f"La imagen tiene una representación no compatible: {imagen.shape}"
        )

    return imagen


def _energia_frecuencial(path_imagen, radio=0.5):
    """Calcula las proporciones de energía alta/baja y sus energías absolutas."""
    imagen = _leer_gris(path_imagen)
    espectro = np.fft.fftshift(np.fft.fft2(imagen))
    potencia = np.abs(espectro) ** 2
    alto, ancho = imagen.shape
    cy, cx = alto // 2, ancho // 2
    yy, xx = np.ogrid[:alto, :ancho]
    radio_corte = radio * min(alto, ancho) / 2.0
    mascara_baja = (yy - cy) ** 2 + (xx - cx) ** 2 <= radio_corte**2
    mascara_alta = ~mascara_baja

    energia_total = float(np.sum(potencia))
    energia_baja = float(np.sum(potencia[mascara_baja]))
    energia_alta = float(np.sum(potencia[mascara_alta]))

    if energia_total <= 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        energia_alta / energia_total,
        energia_baja / energia_total,
        energia_alta,
        energia_baja,
    )


def _obtener_glcm(path_imagen):
    """Construye la GLCM con la configuración utilizada en el proyecto."""
    gris = _leer_gris(path_imagen)
    gris_8bit = img_as_ubyte(gris)
    return graycomatrix(
        gris_8bit,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )


# ============================================================
# CARACTERÍSTICAS: LUMINANCIA Y BRILLO
# ============================================================
def obtener_media_luminancia(img_path):
    return float(np.mean(_leer_gris(img_path)))


def obtener_varianza_luminancia(img_path):
    return float(np.var(_leer_gris(img_path)))


def obtener_rango_dinamico(img_path, epsilon=1e-6):
    imagen = _leer_gris(img_path)
    y_min = float(np.min(imagen))
    y_max = float(np.max(imagen))
    return float(y_max / (y_min + epsilon))


def obtener_desv_estandar_brillo(img_path):
    return float(np.std(_leer_gris(img_path)))


# ============================================================
# CARACTERÍSTICAS: CONTRASTE
# ============================================================
def obtener_contraste_michelson(img_path):
    imagen = _leer_gris(img_path)
    l_max = float(np.percentile(imagen, 98))
    l_min = float(np.percentile(imagen, 2))
    return float((l_max - l_min) / (l_max + l_min + 1e-8))


def obtener_contraste_gradiente(img_path):
    imagen = _leer_gris(img_path)
    return float(np.mean(filters.sobel(imagen)))


# ============================================================
# CARACTERÍSTICAS: ENTROPÍA
# ============================================================
def obtener_entropia_shannon(img_path):
    imagen = img_as_ubyte(_leer_gris(img_path))
    return float(shannon_entropy(imagen))


def obtener_entropia_espectral(img_path):
    imagen = _leer_gris(img_path)
    fft = np.fft.rfft2(imagen)
    potencia = np.abs(fft) ** 2
    suma = float(np.sum(potencia))
    if suma <= 0:
        return float("nan")
    probabilidades = potencia.ravel() / suma
    probabilidades = probabilidades[probabilidades > 0]
    return float(entropy(probabilidades, base=2))


def obtener_entropia_gradiente(img_path):
    imagen = _leer_gris(img_path)
    gradiente = filters.sobel(imagen)
    histograma, _ = np.histogram(gradiente, bins=256)
    total = int(histograma.sum())
    if total == 0:
        return 0.0
    probabilidades = histograma.astype(float) / total
    probabilidades = probabilidades[probabilidades > 0]
    return float(entropy(probabilidades, base=2))


def obtener_entropia_glcm(img_path):
    glcm = _obtener_glcm(img_path)
    probabilidades = np.mean(glcm, axis=(2, 3))
    probabilidades = probabilidades[probabilidades > 0]
    if probabilidades.size == 0:
        return 0.0
    return float(-np.sum(probabilidades * np.log2(probabilidades)))


# ============================================================
# CARACTERÍSTICAS: DOMINIO FRECUENCIAL
# ============================================================
def obtener_energia_altas_frecuencias(img_path, radio=0.5):
    energia_alta, _, _, _ = _energia_frecuencial(img_path, radio=radio)
    return float(energia_alta)


def obtener_energia_bajas_frecuencias(img_path, radio=0.5):
    _, energia_baja, _, _ = _energia_frecuencial(img_path, radio=radio)
    return float(energia_baja)


def obtener_hlfr(img_path, radio=0.5, epsilon=1e-6):
    _, _, energia_alta, energia_baja = _energia_frecuencial(
        img_path, radio=radio
    )
    return float(energia_alta / (energia_baja + epsilon))


# ============================================================
# CARACTERÍSTICAS: BORDES Y GRADIENTES
# ============================================================
def obtener_densidad_bordes(img_path, sigma=1.0):
    imagen = _leer_gris(img_path)
    bordes = feature.canny(imagen, sigma=sigma)
    return float(np.count_nonzero(bordes) / bordes.size)


def obtener_desv_estandar_gradiente(img_path):
    imagen = _leer_gris(img_path)
    gradiente = filters.sobel(imagen)
    return float(np.std(gradiente))


# ============================================================
# CARACTERÍSTICAS: TEXTURA GLCM
# ============================================================
def obtener_homogeneidad_textura(img_path):
    # Se conserva la transformación usada para esta característica en el código
    # descrito en el informe del proyecto.
    gris = _leer_gris(img_path)
    gris_8bit = np.clip(gris * 255.0, 0, 255).astype(np.uint8)
    glcm = graycomatrix(
        gris_8bit,
        distances=[1],
        angles=[0],
        levels=256,
        normed=True,
    )
    return float(graycoprops(glcm, "homogeneity")[0, 0])


def obtener_energia_glcm(img_path, distancias=(1,), angulos=(0,), niveles=256):
    gris = img_as_ubyte(_leer_gris(img_path))
    glcm = graycomatrix(
        gris,
        distances=list(distancias),
        angles=list(angulos),
        levels=niveles,
        symmetric=True,
        normed=True,
    )
    return float(np.mean(graycoprops(glcm, "energy")))


def obtener_correlacion_glcm(img_path):
    glcm = _obtener_glcm(img_path)
    valor = float(np.mean(graycoprops(glcm, "correlation")))
    return valor if np.isfinite(valor) else 0.0


# ============================================================
# CARACTERÍSTICAS: COLOR
# ============================================================
def obtener_media_saturacion(img_path):
    hsv = color.rgb2hsv(cargar_imagen_rgb(img_path))
    return float(np.mean(hsv[:, :, 1]))


def obtener_varianza_saturacion(img_path):
    hsv = color.rgb2hsv(cargar_imagen_rgb(img_path))
    return float(np.var(hsv[:, :, 1]))


def obtener_varianza_matiz(img_path, umbral_saturacion=0.1):
    hsv = color.rgb2hsv(cargar_imagen_rgb(img_path))
    hue = hsv[:, :, 0]
    saturacion = hsv[:, :, 1]
    mascara = saturacion > umbral_saturacion

    if not np.any(mascara):
        return float("nan")

    theta = hue[mascara] * 2.0 * np.pi
    return float(circvar(theta, low=0.0, high=2.0 * np.pi))


# ============================================================
# CARACTERÍSTICAS: RESOLUCIÓN ESPACIAL
# ============================================================
def obtener_megapixeles(img_path):
    imagen = io.imread(str(img_path))
    alto, ancho = imagen.shape[:2]
    return float((alto * ancho) / 1_000_000.0)


def obtener_aspect_ratio(img_path):
    imagen = io.imread(str(img_path))
    alto, ancho = imagen.shape[:2]
    if alto == 0:
        raise ValueError("La imagen tiene altura cero.")
    return float(ancho / alto)


# ============================================================
# EXTRACCIÓN CONSOLIDADA DE CARACTERÍSTICAS
# ============================================================
def extraer_features(path, prefix):
    """Extrae las 26 características de entrada definidas para el proyecto."""
    f = {}
    f[f"{prefix}_media_lum"] = obtener_media_luminancia(path)
    f[f"{prefix}_var_lum"] = obtener_varianza_luminancia(path)
    f[f"{prefix}_rango"] = obtener_rango_dinamico(path)
    f[f"{prefix}_std"] = obtener_desv_estandar_brillo(path)
    f[f"{prefix}_michelson"] = obtener_contraste_michelson(path)

    f[f"{prefix}_hf"] = obtener_energia_altas_frecuencias(path)
    f[f"{prefix}_lf"] = obtener_energia_bajas_frecuencias(path)
    f[f"{prefix}_hlfr"] = obtener_hlfr(path)

    f[f"{prefix}_edges_s1"] = obtener_densidad_bordes(path, sigma=1.0)
    f[f"{prefix}_edges_s2"] = obtener_densidad_bordes(path, sigma=2.0)
    f[f"{prefix}_edges_s3"] = obtener_densidad_bordes(path, sigma=3.0)

    f[f"{prefix}_grad_mean"] = obtener_contraste_gradiente(path)
    f[f"{prefix}_grad_std"] = obtener_desv_estandar_gradiente(path)
    f[f"{prefix}_glcm_contrast"] = float(
        np.mean(graycoprops(_obtener_glcm(path), "contrast"))
    )
    f[f"{prefix}_glcm_hom"] = obtener_homogeneidad_textura(path)
    f[f"{prefix}_glcm_energy"] = obtener_energia_glcm(path)
    f[f"{prefix}_glcm_corr"] = obtener_correlacion_glcm(path)

    f[f"{prefix}_entropia_shannon"] = obtener_entropia_shannon(path)
    f[f"{prefix}_entropia_espectral"] = obtener_entropia_espectral(path)
    f[f"{prefix}_entropia_gradiente"] = obtener_entropia_gradiente(path)
    f[f"{prefix}_entropia_glcm"] = obtener_entropia_glcm(path)

    f[f"{prefix}_sat_mean"] = obtener_media_saturacion(path)
    f[f"{prefix}_sat_var"] = obtener_varianza_saturacion(path)
    f[f"{prefix}_hue_var"] = obtener_varianza_matiz(path)

    f[f"{prefix}_megapixeles"] = obtener_megapixeles(path)
    f[f"{prefix}_aspect_ratio"] = obtener_aspect_ratio(path)
    return f


# ============================================================
# CARGAR ARTEFACTOS UNA SOLA VEZ
# ============================================================
def cargar_artefactos():
    global MODELO, PREPROCESAMIENTO, CONFIGURACION

    if MODELO is not None and PREPROCESAMIENTO is not None:
        return MODELO, PREPROCESAMIENTO, CONFIGURACION

    rutas = [RUTA_PREPROCESAMIENTO, RUTA_MODELO, RUTA_CONFIGURACION]
    faltantes = [str(ruta) for ruta in rutas if not ruta.exists()]
    if faltantes:
        raise FileNotFoundError(
            "No se encontraron los siguientes artefactos del modelo:\n\n"
            + "\n".join(faltantes)
            + "\n\nColoca las carpetas 'modelo_lpips_exportado' y "
              "'resultados_optimizacion' junto a este archivo .py."
        )

    PREPROCESAMIENTO = joblib.load(RUTA_PREPROCESAMIENTO)
    MODELO = tf.keras.models.load_model(RUTA_MODELO)
    with open(RUTA_CONFIGURACION, "r", encoding="utf-8") as archivo:
        CONFIGURACION = json.load(archivo)

    return MODELO, PREPROCESAMIENTO, CONFIGURACION


# ============================================================
# PREPARAR UNA IMAGEN PARA LA INFERENCIA
# ============================================================
def preparar_entrada(path_imagen, formato_salida, metodo, calidad):
    """Prepara las variables y verifica que la entrada cumpla los requisitos."""
    info_imagen = inspeccionar_imagen(path_imagen)
    path_imagen = info_imagen["ruta"]
    formato_entrada = info_imagen["formato"]

    formato_salida = str(formato_salida).lower().strip().lstrip(".")
    if formato_salida == "jpeg":
        formato_salida = "jpg"
    if formato_salida not in FORMATOS_VALIDOS:
        raise ValueError(f"Formato de salida no admitido: {formato_salida}")
    if formato_salida == formato_entrada:
        opciones = sorted(FORMATOS_VALIDOS - {formato_entrada})
        raise ValueError(
            f"El formato de salida debe ser distinto al formato original ({formato_entrada.upper()}). "
            f"Formatos de salida permitidos: {', '.join(opciones).upper()}."
        )

    if not isinstance(calidad, (int, np.integer)):
        raise TypeError("La calidad debe ser un número entero.")
    if int(calidad) not in CALIDADES_VALIDAS:
        raise ValueError(
            "La calidad debe ser uno de estos niveles: 10, 30, 50, 70 o 90."
        )

    _, preprocesamiento, _ = cargar_artefactos()
    caracteristicas = extraer_features(str(path_imagen), prefix="orig")
    X = pd.DataFrame([caracteristicas])

    columnas_eliminar = [
        "orig_var_lum",
        "orig_edges_s1",
        "orig_edges_s2",
        "orig_hlfr",
        "orig_rango",
    ]
    X = X.drop(columns=columnas_eliminar, errors="ignore")

    if "orig_hue_var" not in X.columns:
        raise ValueError("La extracción no generó la característica orig_hue_var.")

    X["orig_hue_var_faltante"] = X["orig_hue_var"].isna().astype(int)
    X["orig_hue_var"] = X["orig_hue_var"].fillna(
        preprocesamiento["mediana_hue"]
    )

    X["formato_entrada"] = formato_entrada
    X["formato_salida"] = formato_salida
    X["metodo"] = metodo
    X["calidad"] = int(calidad)

    columnas_categoricas = preprocesamiento["columnas_categoricas"]
    columnas_modelo = preprocesamiento["columnas_modelo"]

    for columna in columnas_categoricas:
        if columna not in X.columns:
            raise ValueError(f"Falta la columna categórica requerida: {columna}")
        valor = X[columna].iloc[0]
        prefijo = f"{columna}_"
        categorias_conocidas = {
            nombre[len(prefijo):]
            for nombre in columnas_modelo
            if nombre.startswith(prefijo)
        }
        if valor not in categorias_conocidas:
            raise ValueError(
                f"Categoría no reconocida para '{columna}': '{valor}'. "
                f"Categorías disponibles: {sorted(categorias_conocidas)}"
            )

    X = pd.get_dummies(X, columns=columnas_categoricas, dtype=int)
    X = X.reindex(columns=columnas_modelo, fill_value=0)

    columnas_continuas = preprocesamiento["columnas_continuas"]
    # Convertir a float antes del escalado evita la advertencia de pandas
    # por asignar resultados decimales sobre columnas inicialmente enteras.
    X[columnas_continuas] = preprocesamiento["scaler"].transform(
        X[columnas_continuas].astype(float)
    )

    if X.isna().any().any():
        columnas_nulas = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Quedaron valores faltantes en la entrada: {columnas_nulas}")
    if X.columns.tolist() != columnas_modelo:
        raise ValueError("Las columnas no coinciden con las del entrenamiento.")

    return X, formato_entrada, formato_salida


# ============================================================
# PREDICCIÓN ANTICIPADA DE LPIPS
# ============================================================
def predecir_lpips(path_imagen, formato_salida, metodo, calidad):
    """Estima LPIPS antes de recodificar y devuelve metadatos de la predicción."""
    modelo, _, _ = cargar_artefactos()
    X, formato_entrada, formato_salida = preparar_entrada(
        path_imagen=path_imagen,
        formato_salida=formato_salida,
        metodo=metodo,
        calidad=calidad,
    )

    prediccion_original = float(modelo.predict(X, verbose=0).reshape(-1)[0])
    prediccion_no_negativa = max(0.0, prediccion_original)

    return {
        "lpips_estimado_original": prediccion_original,
        "lpips_estimado_no_negativo": prediccion_no_negativa,
        "prediccion_negativa": prediccion_original < 0,
        "formato_entrada": formato_entrada,
        "formato_salida": formato_salida,
        "metodo": metodo,
        "calidad": int(calidad),
    }


# ============================================================
# INTERFAZ GRÁFICA MÍNIMA (TKINTER)
# ============================================================
def abrir_interfaz():
    ventana = tk.Tk()
    ventana.title("Estimador de pérdida perceptual (LPIPS)")
    ventana.geometry("560x570")
    ventana.minsize(520, 540)
    ventana.resizable(False, False)

    ruta_imagen = tk.StringVar(value="")
    formato = tk.StringVar(value="")
    metodo = tk.StringVar(value="pillow")
    calidad = tk.StringVar(value="50")
    estado = tk.StringVar(value="Selecciona una imagen para comenzar.")

    estilo = ttk.Style(ventana)
    try:
        estilo.theme_use("clam")
    except tk.TclError:
        pass

    contenedor = ttk.Frame(ventana, padding=22)
    contenedor.pack(fill="both", expand=True)

    ttk.Label(
        contenedor,
        text="Estimador de calidad perceptual",
        font=("Segoe UI", 16, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        contenedor,
        text="Predicción de LPIPS antes de realizar la recodificación.",
        wraplength=490,
    ).pack(anchor="w", pady=(5, 15))

    grupo_imagen = ttk.LabelFrame(contenedor, text="Imagen original", padding=12)
    grupo_imagen.pack(fill="x", pady=(0, 12))

    fila_imagen = ttk.Frame(grupo_imagen)
    fila_imagen.pack(fill="x")
    nombre_imagen = ttk.Label(
        fila_imagen,
        text="Ninguna imagen seleccionada",
        wraplength=330,
    )
    nombre_imagen.pack(side="left", fill="x", expand=True, padx=(0, 8))

    info_imagen_label = ttk.Label(
        grupo_imagen,
        text="Requisitos: RGB y resolución mínima de 64 × 64 píxeles.",
        wraplength=480,
        foreground="#555555",
    )
    info_imagen_label.pack(anchor="w", pady=(8, 0))

    grupo_parametros = ttk.LabelFrame(
        contenedor, text="Transformación prevista", padding=12
    )
    grupo_parametros.pack(fill="x", pady=(0, 12))
    grupo_parametros.columnconfigure(1, weight=1)

    ttk.Label(grupo_parametros, text="Formato de salida:").grid(
        row=0, column=0, sticky="w", padx=(0, 12), pady=5
    )
    combo_formato = ttk.Combobox(
        grupo_parametros,
        textvariable=formato,
        values=[],
        state="disabled",
        width=18,
    )
    combo_formato.grid(row=0, column=1, sticky="w", pady=5)

    ttk.Label(grupo_parametros, text="Método:").grid(
        row=1, column=0, sticky="w", padx=(0, 12), pady=5
    )
    combo_metodo = ttk.Combobox(
        grupo_parametros,
        textvariable=metodo,
        values=["pillow", "imagemagick", "ffmpeg"],
        state="readonly",
        width=18,
    )
    combo_metodo.grid(row=1, column=1, sticky="w", pady=5)

    ttk.Label(grupo_parametros, text="Nivel de calidad:").grid(
        row=2, column=0, sticky="w", padx=(0, 12), pady=5
    )
    combo_calidad = ttk.Combobox(
        grupo_parametros,
        textvariable=calidad,
        values=["10", "30", "50", "70", "90"],
        state="readonly",
        width=18,
    )
    combo_calidad.grid(row=2, column=1, sticky="w", pady=5)

    grupo_resultado = ttk.LabelFrame(contenedor, text="Resultado", padding=12)
    grupo_resultado.pack(fill="x", pady=(0, 12))

    valor_resultado = ttk.Label(
        grupo_resultado,
        text="LPIPS estimado: —",
        font=("Segoe UI", 14, "bold"),
    )
    valor_resultado.pack(anchor="w", pady=(0, 7))

    detalles = ttk.Label(grupo_resultado, text="", justify="left", wraplength=490)
    detalles.pack(anchor="w")

    def seleccionar_imagen():
        archivo = filedialog.askopenfilename(
            parent=ventana,
            title="Seleccionar imagen original",
            filetypes=[("Imágenes compatibles", "*.jpg *.jpeg *.png *.webp")],
        )
        if not archivo:
            return

        ruta_imagen.set(archivo)
        nombre_imagen.config(text=Path(archivo).name)
        valor_resultado.config(text="LPIPS estimado: —")
        detalles.config(text="")
        boton_estimar.config(state="disabled")
        combo_formato.config(values=[], state="disabled")
        formato.set("")

        try:
            info = inspeccionar_imagen(archivo)
        except Exception as exc:
            info_imagen_label.config(
                text=f"Imagen no válida: {exc}", foreground="#b42318"
            )
            estado.set("Predicción bloqueada: la imagen no cumple los requisitos.")
            messagebox.showwarning(
                "Imagen no válida",
                "No se puede estimar LPIPS con esta imagen.\n\n" + str(exc),
                parent=ventana,
            )
            return

        formato_entrada = info["formato"]
        # Solo se ofrecen los otros dos formatos, nunca el formato original.
        formatos_salida = [f for f in ["jpg", "png", "webp"] if f != formato_entrada]
        combo_formato.config(values=formatos_salida, state="readonly")
        formato.set("webp" if "webp" in formatos_salida else formatos_salida[0])

        info_imagen_label.config(
            text=(
                f"Validación correcta: {info['ancho']} × {info['alto']} px | "
                f"Modo {info['modo']} | Formato {formato_entrada.upper()}."
            ),
            foreground="#18794e",
        )
        boton_estimar.config(state="normal")
        estado.set(
            f"Imagen válida. Formatos de salida disponibles: "
            f"{', '.join(formato.upper() for formato in formatos_salida)}."
        )

    ttk.Button(fila_imagen, text="Buscar…", command=seleccionar_imagen).pack(
        side="right"
    )

    def terminar_prediccion(resultado=None, error=None):
        if error is not None:
            # Si el error corresponde a un requisito de entrada, vuelve a bloquear.
            try:
                inspeccionar_imagen(ruta_imagen.get())
                entrada_valida = True
            except Exception:
                entrada_valida = False
            boton_estimar.config(state="normal" if entrada_valida else "disabled")
            estado.set("La predicción no pudo completarse.")
            messagebox.showerror("Error en la predicción", str(error), parent=ventana)
            return

        boton_estimar.config(state="normal")
        valor_resultado.config(
            text=f"LPIPS estimado: {resultado['lpips_estimado_no_negativo']:.6f}"
        )
        aviso = (
            "\nAdvertencia: la salida original del modelo fue negativa y se mostró limitada a cero."
            if resultado["prediccion_negativa"]
            else ""
        )
        detalles.config(
            text=(
                f"Formato original: {resultado['formato_entrada']}\n"
                f"Formato de salida: {resultado['formato_salida']}\n"
                f"Método previsto: {resultado['metodo']}\n"
                f"Calidad seleccionada: {resultado['calidad']}"
                f"{aviso}"
            )
        )
        estado.set("Predicción completada. La imagen original no fue modificada.")

    def ejecutar_prediccion():
        archivo = ruta_imagen.get().strip()
        if not archivo:
            boton_estimar.config(state="disabled")
            messagebox.showwarning(
                "Falta la imagen", "Primero selecciona una imagen.", parent=ventana
            )
            return

        # Revalidar justo antes de predecir, aunque la selección ya se validó.
        try:
            info = inspeccionar_imagen(archivo)
        except Exception as exc:
            boton_estimar.config(state="disabled")
            estado.set("Predicción bloqueada: la imagen no cumple los requisitos.")
            messagebox.showwarning("Imagen no válida", str(exc), parent=ventana)
            return

        formato_salida = formato.get()
        if not formato_salida or formato_salida == info["formato"]:
            boton_estimar.config(state="disabled")
            messagebox.showwarning(
                "Formato de salida no válido",
                "Selecciona uno de los otros dos formatos, distinto al original.",
                parent=ventana,
            )
            return

        metodo_seleccionado = metodo.get()
        calidad_seleccionada = int(calidad.get())

        boton_estimar.config(state="disabled")
        valor_resultado.config(text="Calculando…")
        detalles.config(text="Extrayendo características y ejecutando el modelo.")
        estado.set("Procesando imagen; espera un momento…")

        def trabajo():
            try:
                resultado = predecir_lpips(
                    path_imagen=archivo,
                    formato_salida=formato_salida,
                    metodo=metodo_seleccionado,
                    calidad=calidad_seleccionada,
                )
                ventana.after(0, lambda r=resultado: terminar_prediccion(resultado=r))
            except Exception as exc:
                ventana.after(0, lambda e=exc: terminar_prediccion(error=e))

        threading.Thread(target=trabajo, daemon=True).start()

    boton_estimar = ttk.Button(
        contenedor, text="Estimar LPIPS", command=ejecutar_prediccion, state="disabled"
    )
    boton_estimar.pack(fill="x", ipady=4, pady=(0, 8))

    ttk.Label(
        contenedor,
        textvariable=estado,
        foreground="#555555",
        wraplength=490,
    ).pack(anchor="w")

    ventana.mainloop()


# ============================================================
# INICIO DEL PROGRAMA
# ============================================================
if __name__ == "__main__":
    abrir_interfaz()
