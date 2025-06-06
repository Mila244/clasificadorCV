import streamlit as st
import pdfplumber
import re
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Clasificador de Currículos - Centro de Salud",
    page_icon="🩺",
    layout="centered"
)

# --- LOGO Y ENCABEZADO ---
st.image("logo.png", width=150)
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Clasificador de Currículos</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Centro de Salud – Selección por Especialidad</h4>", unsafe_allow_html=True)
st.write("---")

# --- FUNCION DE PREPROCESAMIENTO ---
def limpiar_texto(texto):
    texto = texto.lower()  # pasar a minúsculas
    texto = re.sub(r'\n', ' ', texto)  # quitar saltos de línea
    texto = re.sub(r'[^\w\s]', ' ', texto)  # eliminar caracteres especiales
    texto = re.sub(r'\s+', ' ', texto)  # eliminar espacios extras
    return texto.strip()

# --- DATOS DE ENTRENAMIENTO MEJORADOS ---
textos = [
    # Obstetricia
    "control prenatal atención parto planificación familiar salud materna ginecología obstétrica embarazo parto natural ecografías atención gestantes",
    "parto humanizado monitoreo fetal cuidados pre y postnatales salud materna programas comunitarios embarazo seguro lactancia materna",
    
    # Odontología
    "salud bucal limpieza dental extracciones caries ortodoncia odontopediatría radiografías dentales atención en consultorio tratamiento dental",
    "prevención caries tratamiento conducto prótesis dentales cirugía oral odontología estética blanqueamiento limpieza profunda",
    
    # Laboratorio
    "análisis clínicos muestras sangre hemograma pruebas serológicas bioquímica clínica procesamiento muestras microscopio diagnóstico exámenes médicos",
    "laboratorio clínico control calidad toma de muestras análisis microbiológico pruebas hematológicas química clínica",
    
    # Enfermería
    "cuidado pacientes administración medicamentos monitoreo signos vitales primeros auxilios inyectables higiene post operatoria enfermería asistencial",
    "atención directa pacientes cuidados intensivos administración de medicamentos control signos vitales asistencia post operatoria",
    
    # Doctor / Médico
    "diagnóstico médico consulta externa recetas emergencias atención primaria historial clínico medicina general prescripción tratamientos urgencias",
    "consulta médica evaluación clínica tratamiento diagnóstico especialista historia clínica atención hospitalaria",
    
    # Administración y Estadística
    "gestión personal informes estadísticos administración recursos atención público procesamiento datos excel reportes médicos archivo clínico sistema estadístico",
    "administración hospitalaria coordinación personal manejo presupuestos informes estadísticos análisis datos bases de datos",
    
    # Limpieza
    "limpieza hospitalaria desinfección ambientes recolección residuos aseo consultorios manejo productos limpieza protocolos bioseguridad limpieza salas médicas",
    "desinfección áreas hospitalarias aseo limpieza mantenimiento higiene control infecciones manejo químicos productos limpieza hospital"
]

areas = [
    "Obstetricia", "Obstetricia",
    "Odontología", "Odontología",
    "Laboratorio", "Laboratorio",
    "Enfermería", "Enfermería",
    "Doctor", "Doctor",
    "Administración y Estadística", "Administración y Estadística",
    "Limpieza", "Limpieza"
]

# --- ENTRENAMIENTO DEL MODELO ---
modelo = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), max_df=0.85, min_df=1),
    LogisticRegression(max_iter=1000, random_state=42)
)
modelo.fit(textos, areas)

# --- ESTILO PERSONALIZADO PARA LA CARGA DE ARCHIVO ---
st.markdown("""
    <style>
    .css-1y4p8pa, .css-1uixxvy, .e1b2p2ww0 {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 📄 Arrastra y suelta tu archivo aquí")
st.markdown("_Límite: 200MB por archivo • Formato: PDF_")

# --- CARGA DE ARCHIVO ---
archivo = st.file_uploader("Selecciona un archivo", type="pdf", label_visibility="collapsed")

# --- PROCESAMIENTO Y CLASIFICACIÓN ---
if archivo is not None:
    with pdfplumber.open(archivo) as pdf:
        texto = ""
        for pagina in pdf.pages:
            contenido = pagina.extract_text()
            if contenido:
                texto += contenido + "\n"

    texto = limpiar_texto(texto)

    if texto != "":
        if st.button("🔍 Clasificar CV"):
            prob = modelo.predict_proba([texto])[0]
            pred = modelo.classes_[prob.argmax()]
            max_prob = prob.max()

            if max_prob >= 0.5:
                st.success(f"🗂️ Área sugerida para el candidato: **{pred}** (Confianza: {max_prob*100:.2f}%)")
            else:
                st.warning("⚠️ No se pudo clasificar con suficiente confianza. Por favor revise el CV o mejore el modelo.")

            st.markdown("### 📊 Detalle de Probabilidades")
            for label, p in zip(modelo.classes_, prob):
                st.write(f"- **{label}**: {p*100:.2f}%")
    else:
        st.warning("⚠️ No se pudo extraer texto del PDF. Asegúrate de que el archivo no esté escaneado como imagen.")
