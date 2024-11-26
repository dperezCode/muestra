import cv2
import streamlit as st
from ultralytics import YOLO
import os
from fpdf import FPDF

CARPETA_CARGA = 'uploads'

# Función para procesar la imagen y generar detecciones
def procesar_imagen(imagen_path, modelo):
    imagen = cv2.imread(imagen_path)
    imagen = cv2.resize(imagen, (340, 340))  
    resultados = modelo.predict(imagen, conf=0.2)  
    detecciones = resultados[0].boxes.data.cpu().numpy()

    plagas_detectadas = []
    for box in detecciones:
        x1, y1, x2, y2, conf, class_id = box
        class_id = int(class_id)
        plaga = modelo.names[class_id]
        exactitud = round(conf * 100, 2)  
        plagas_detectadas.append((plaga, exactitud, (x1, y1, x2, y2)))

        centro_x = int((x1 + x2) / 2)
        centro_y = int((y1 + y2) / 2)
        radio = int(max(x2 - x1, y2 - y1) / 4)
        cv2.circle(imagen, (centro_x, centro_y), radio, (0, 0, 255), 2)

    imagen_procesada_path = os.path.splitext(imagen_path)[0] + "_procesada.jpg"
    cv2.imwrite(imagen_procesada_path, imagen)

    return plagas_detectadas, imagen_procesada_path

# Función para generar el PDF con las imágenes procesadas y sus resultados
def generar_pdf(resultados_imagenes):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="Resultados de la Detección de Plagas en Tomate", ln=True, align='C')

    for imagen_path, detecciones in resultados_imagenes:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 10, txt=f"Resultados para: {os.path.basename(imagen_path)}", ln=True, align='C')

        for plaga, exactitud, _ in detecciones:
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, txt=f"Plaga: {plaga}, Exactitud: {exactitud}%", ln=True)

        pdf.image(imagen_path, x=10, y=pdf.get_y(), w=100)
        pdf.ln(70)

    output_pdf = os.path.join(CARPETA_CARGA, "resultados_plagas.pdf")
    pdf.output(output_pdf)

    return output_pdf

def main():
    st.set_page_config(page_title="Detección de Plagas en Tomate", page_icon="🍅", layout="wide")
    st.markdown("<h1 style='text-align: center;'>Plaga Scan 🍅  </h1>", unsafe_allow_html=True)

    modelo = YOLO("tomate2.pt")

    descripcion = """
    <p style='font-size: 20px; text-align: justify;'>
        <b> Siga los siguientes pasos</b> .
        <ol style='font-size: 18px;'>
             <li><b>Cargar solo imágenes de hojas de tomate para determinar si se encuentra afectada por la plaga de tuta absoluta:</b></li>
            <li><b>Carga de imágenes:</b> Puede cargar una o varias imágenes seleccionándolas desde su dispositivo.</li>
            <li><b>Análisis de imágenes:</b> Se mostrarán las imágenes analizadas con las plagas detectadas y su índice de similitud.</li>
            <li><b>Generación de PDF:</b> Puede descargar un PDF con los resultados e imágenes procesadas.</li>
        </ol>
    </p>
    """
    st.write(descripcion, unsafe_allow_html=True)
    archivos_imagenes = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if archivos_imagenes:
        resultados_imagenes = []
        for archivo in archivos_imagenes:
            ruta_imagen = os.path.join(CARPETA_CARGA, archivo.name)
            with open(ruta_imagen, "wb") as f:
                f.write(archivo.getvalue())

            detecciones, imagen_procesada_path = procesar_imagen(ruta_imagen, modelo)
            resultados_imagenes.append((imagen_procesada_path, detecciones))

        # Mostrar las imágenes procesadas con índices de similitud
        for imagen_procesada_path, detecciones in resultados_imagenes:
            st.image(imagen_procesada_path, caption="Imagen Procesada", use_column_width=True)
            for plaga, exactitud, _ in detecciones:
                st.write(f"**Plaga:** {plaga}, **Exactitud:** {exactitud}%")

        # Generar el PDF y mostrar el botón de descarga
        if resultados_imagenes:
            pdf_path = generar_pdf(resultados_imagenes)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Descargar PDF de Resultados",
                    data=f,
                    file_name="resultados_plagas.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
