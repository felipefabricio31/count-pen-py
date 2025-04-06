import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from collections import Counter

st.set_page_config(page_title="Detector de Canetinhas", layout="centered")
st.title("🖍️ Detector de Canetinhas Touch")
st.write("Faça upload de uma imagem da bolsa para contar as canetinhas e verificar se há repetidas.")

# --- Configuração Tesseract ---
custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# --- Funções de processamento ---

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    return thresh

def extract_labels(image_cv):
    processed = preprocess_for_ocr(image_cv)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 30 < w < 120 and 30 < h < 120:  # tamanho típico de uma tampa
            roi = image_cv[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (100, 100))
            text = pytesseract.image_to_string(roi_gray, config=custom_config).strip()
            if 2 <= len(text) <= 6:
                labels.append(text)

    return labels

def identificar_tipo_bolsa(qtd):
    tipos = [12, 24, 36, 48, 60, 80, 120]
    mais_proximo = min(tipos, key=lambda x: abs(x - qtd))
    return mais_proximo

# --- Interface de Upload ---

imagem = st.file_uploader("📁 Envie uma imagem", type=["png", "jpg", "jpeg"])

if imagem:
    img_pil = Image.open(imagem).convert("RGB")
    st.image(img_pil, caption="Imagem enviada", use_column_width=True)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("🔎 Analisando canetinhas..."):
        etiquetas = extract_labels(img_cv)
        total = len(etiquetas)
        repetidas = {k: v for k, v in Counter(etiquetas).items() if v > 1}
        tipo_estimado = identificar_tipo_bolsa(total)

    st.success(f"🖊️ Total de canetinhas detectadas: **{total}**")
    st.info(f"📦 Provavelmente é uma bolsa de **{tipo_estimado} canetinhas**")

    if repetidas:
        st.warning("🔁 Canetinhas repetidas detectadas:")
        for etiqueta, qtd in repetidas.items():
            st.write(f"- `{etiqueta}` apareceu {qtd} vezes")
    else:
        st.success("✅ Nenhuma canetinha repetida encontrada.")
