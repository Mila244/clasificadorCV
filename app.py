import streamlit as st
import pdfplumber
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords

# Descargar las stopwords si no las tienes (solo necesitas hacerlo una vez)
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Clasificador de Currículos - Centro de Salud",
    page_icon="🩺",
    layout="centered"
)

# --- LOGO Y ENCABEZADO ---
# Asegúrate de tener 'logo.png' en la misma carpeta o proporciona la ruta correcta.
st.image("logo.png", width=150)
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Clasificador de Currículos</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Centro de Salud – Selección por Especialidad</h4>", unsafe_allow_html=True)
st.write("---")

# --- FUNCION DE PREPROCESAMIENTO MEJORADA ---
def limpiar_texto(texto):
    texto = texto.lower()  # pasar a minúsculas
    texto = re.sub(r'\n', ' ', texto)  # quitar saltos de línea
    texto = re.sub(r'[^\w\s]', ' ', texto)  # eliminar caracteres especiales
    texto = re.sub(r'\d+', '', texto) # Eliminar números
    texto = re.sub(r'\s+', ' ', texto)  # eliminar espacios extras
    
    # Eliminar stop words
    stop_words = set(stopwords.words('spanish'))
    texto_sin_stopwords = [word for word in texto.split() if word not in stop_words]
    return " ".join(texto_sin_stopwords).strip()

# --- DATOS DE ENTRENAMIENTO MEJORADOS Y EXPANDIDOS ---
# Es crucial tener muchos más ejemplos y más diversos para cada categoría.
# Estos son solo ejemplos, idealmente deberías tener MUCHOS más de cada uno.
textos = [
    # Obstetricia (Ejemplos expandidos y más detallados)
    "control prenatal atención parto planificación familiar salud materna ginecología obstétrica embarazo parto natural ecografías atención gestantes monitoreo fetal cuidados pre y postnatales embarazo seguro lactancia materna educación prenatal urgencias obstétricas seguimiento del embarazo partería salud reproductiva puérperas bienestar materno-infantil matrona ecografía 4D consejería salud sexual",
    "parto humanizado monitoreo fetal cuidados pre y postnatales programas comunitarios embarazo seguro lactancia materna atención sala partos manejo complicaciones parto asistencia alumbramiento cesáreas atención mujer embarazada control peso gestacional detección riesgos embarazo planificación familiar postparto",
    "atención integral embarazo seguimiento gestación parto vaginal control hemorragias postparto cuidado recién nacido salud ginecológica métodos anticonceptivos infertilidad salud mujer consulta obstétrica educación para parto curso psicoprofiláctico visita domiciliaria gestacional preeclampsia diabetes gestacional",
    "servicio obstetricia especialista obstetricia salud femenina cuidados durante embarazo trabajo parto alumbramiento detección temprana anomalías fetales ultrasonido obstétrico parto agua doula recuperación postparto manejo dolor parto apoyo lactancia consejería familiar salud sexual",
    "ginecólogo obstetra control gestacional salud reproductiva parto sin dolor ecografía 3D atención postparto enfermería obstétrica planificación familiar",
    "matrona partera asistencia alumbramiento cuidados madre e hijo educación maternal puericultura lactancia materna",

    # Odontología (Ejemplos expandidos y más detallados)
    "salud bucal limpieza dental extracciones caries ortodoncia odontopediatría radiografías dentales atención consultorio tratamiento dental endodoncia periodoncia prótesis dentales implantología estética dental blanqueamiento restauración dental operatoria dental cirugía oral frenectomía exodoncias urgencias dentales",
    "prevención caries tratamiento conducto prótesis dentales cirugía oral odontología estética blanqueamiento limpieza profunda profilaxis dental sellantes flúor empastes coronas puentes carillas recontorneado gingival ortodoncia invisible Invisalign brackets diagnóstico oral patología bucal manejo dolor dental",
    "diagnóstico tratamiento enfermedades bucales higiene oral educación dental odontología general atención integral urgencias odontológicas manejo ansiedad pacientes odontología conservadora pulpectomías apicectomías oclusión férulas descarga rehabilitación oral sedación consciente odontología láser",
    "clínica dental especialista odontología salud oral cuidado encías periodontitis gingivitis restauraciones amalgama resina extracciones simples complejas implantes dentales injertos óseos odontología forense traumatismos dentales oclusión articulación temporomandibular ATM odontología digital",
    "estomatólogo odontólogo especialista dental dentista prevención dental ortodoncista periodoncista implantólogo",
    "limpieza sarro curetaje sellado resina dental estética blanqueamiento dental extracciones muelas tratamiento endodoncia",

    # Laboratorio (Ejemplos expandidos y más detallados)
    "análisis clínicos muestras sangre hemograma pruebas serológicas bioquímica clínica procesamiento muestras microscopio diagnóstico exámenes médicos laboratorio clínico control calidad toma muestras análisis microbiológico pruebas hematológicas química clínica urianálisis coprocultivo inmunoensayos",
    "laboratorio clínico control calidad toma de muestras análisis microbiológico pruebas hematológicas química clínica patología clínica citometría flujo biología molecular resultados laboratorio calibración equipos seguridad laboratorio gestión muestras",
    "técnico laboratorio clínico bioquímico biomédico análisis sanguíneo análisis orina pruebas hormonales pruebas genéticas manejo equipos laboratorio automatizados",
    "diagnóstico in vitro procesamiento tejido preparación muestras tinción histopatología citología banco sangre transfusiones hematología",
    "microbiología virología parasitología cultivo bacterias antibiograma control infecciones higiene hospitalaria",
    "laboratorio investigación desarrollo test rápidos pruebas moleculares pcr elisa instrumentación analítica cromatografía",

    # Enfermería (Ejemplos expandidos y más detallados)
    "cuidado pacientes administración medicamentos monitoreo signos vitales primeros auxilios inyectables higiene post operatoria enfermería asistencial atención directa pacientes cuidados intensivos administración medicamentos control signos vitales asistencia post operatoria curaciones vendajes registro médico educación paciente",
    "enfermera titulada enfermero cuidados paliativos atención domiciliaria urgencias enfermería triaje manejo heridas soporte vital básico avanzado registro de enfermería plan de cuidados higiene y confort del paciente",
    "salud pública enfermería comunitaria vacunación campañas de salud promoción salud educación sanitaria prevención enfermedades asistencia primaria",
    "supervisión personal enfermería gestión unidades enfermería evaluación calidad atención seguridad paciente farmaco-vigilancia",
    "cuidados críticos unidad cuidados intensivos UCI monitoreo invasivo ventilación mecánica reanimación cardiopulmonar RCP soporte vital avanzado",
    "enfermera pediátrica neonatología cuidados neonatales atención al niño vacunas infantiles salud escolar",

    # Doctor / Médico (Ejemplos expandidos y más detallados)
    "diagnóstico médico consulta externa recetas emergencias atención primaria historial clínico medicina general prescripción tratamientos urgencias consulta médica evaluación clínica tratamiento diagnóstico especialista historia clínica atención hospitalaria interconsulta",
    "médico general médico especialista internista pediatra cirujano cardiólogo dermatólogo traumatólogo neurólogo psiquiatra radiólogo oncólogo urólogo",
    "diagnóstico diferencial manejo enfermedades crónicas medicina preventiva telemedicina lectura exámenes clínicos interpretación resultados pruebas diagnósticas ética médica atención integral",
    "historia clínica anamnesis exploración física diagnóstico tratamiento pronóstico seguimiento paciente referencia a especialista medicina familiar",
    "urgencias médicas emergencias reanimación triaje atención traumatismos politraumatizados quemaduras intoxicaciones paro cardiorrespiratorio",
    "investigación clínica ensayo clínico publicaciones científicas docencia médica congresos médicos desarrollo profesional continuo",

    # Administración y Estadística (Ejemplos expandidos y más detallados)
    "gestión personal informes estadísticos administración recursos atención público procesamiento datos excel reportes médicos archivo clínico sistema estadístico administración hospitalaria coordinación personal manejo presupuestos informes estadísticos análisis datos bases de datos facturación médica gestión suministros control inventarios",
    "administrador de salud gestión clínica analista de datos hospitalarios estadísticas de salud gestión de proyectos sanitarios sistemas de información de salud SIS recursos humanos contabilidad médica finanzas hospitalarias auditoría médica",
    "manejo de office word excel powerpoint tablas dinámicas bases de datos access spss r python para análisis estadístico minería de datos big data salud pública epidemiología análisis demográfico",
    "secretaria administrativa asistente de gerencia facturación gestión de citas atención telefónica archivo documental gestión de expedientes digitalización de documentos manejo de agenda",
    "calidad en salud acreditación hospitalaria normativas sanitarias auditoría interna gestión de riesgos atención al cliente mejora continua gestión de procesos administración financiera",
    "planificación estratégica hospitalaria análisis de costos evaluación de servicios de salud políticas de salud regulaciones sanitarias gestión de la calidad total ISO sistemas de gestión",

    # Limpieza (Ejemplos expandidos y más detallados)
    "limpieza hospitalaria desinfección ambientes recolección residuos aseo consultorios manejo productos limpieza protocolos bioseguridad limpieza salas médicas desinfección áreas hospitalarias aseo limpieza mantenimiento higiene control infecciones manejo químicos productos limpieza hospital",
    "personal de limpieza operario de limpieza auxiliar de limpieza mantenimiento de instalaciones saneamiento ambiental manejo de desechos biológicos limpieza profunda esterilización de equipos lavandería hospitalaria control de plagas",
    "manejo de maquinaria de limpieza aspiradoras pulidoras barredoras fregadoras uso de equipos de protección personal EPP normas de seguridad industrial limpieza de quirófanos áreas estériles limpieza terminal",
    "manejo de detergentes desinfectantes químicos de limpieza clasificación de residuos hospitalarios seguridad laboral prevención de accidentes trabajo en equipo limpieza de áreas comunes oficinas baños cocinas",
    "limpiador conserje operario higiene encargado limpieza desinfección hospitalaria eliminación residuos limpieza especializada",
    "protocolos higiene manejo desechos químicos bioseguridad ambiental aseo hospitalario control vectores salubridad"
]

areas = [
    "Obstetricia", "Obstetricia", "Obstetricia", "Obstetricia", "Obstetricia", "Obstetricia",
    "Odontología", "Odontología", "Odontología", "Odontología", "Odontología", "Odontología",
    "Laboratorio", "Laboratorio", "Laboratorio", "Laboratorio", "Laboratorio", "Laboratorio",
    "Enfermería", "Enfermería", "Enfermería", "Enfermería", "Enfermería", "Enfermería",
    "Doctor", "Doctor", "Doctor", "Doctor", "Doctor", "Doctor",
    "Administración y Estadística", "Administración y Estadística", "Administración y Estadística", "Administración y Estadística", "Administración y Estadística", "Administración y Estadística",
    "Limpieza", "Limpieza", "Limpieza", "Limpieza", "Limpieza", "Limpieza"
]

# --- ENTRENAMIENTO DEL MODELO CON OPTIMIZACIÓN (GRIDSEARCHCV) ---
# Definimos el pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression(max_iter=2000, random_state=42, solver='liblinear')) # Aumentamos max_iter y elegimos solver
])

# Definimos los parámetros a probar para GridSearchCV
# Experimenta con estos valores para encontrar los mejores para tus datos.
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)], # Probar con unigramas y bigramas
    'tfidf__max_df': [0.7, 0.85, 0.95],   # Ignorar términos que aparecen en demasiados documentos
    'tfidf__min_df': [1, 2],            # Ignorar términos que aparecen en muy pocos documentos
    'logreg__C': [0.1, 1, 10],          # Parámetro de regularización de Logistic Regression
    'logreg__penalty': ['l1', 'l2']     # Tipo de regularización
}

st.write("Entrenando el modelo, por favor espere...")
# Ejecutamos GridSearchCV
# cv=5 significa 5-fold cross-validation
# n_jobs=-1 usa todos los núcleos disponibles de la CPU
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=0)
grid_search.fit(textos, areas)

# El mejor estimador encontrado por GridSearchCV será nuestro modelo final
modelo = grid_search.best_estimator_
st.write(f"Modelo entrenado con los mejores parámetros: **{grid_search.best_params_}**")
st.write(f"Mejor puntuación de validación cruzada: **{grid_search.best_score_:.4f}**")
st.write("---")


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
    with st.spinner('Extrayendo texto del PDF...'):
        texto_extraido = ""
        try:
            with pdfplumber.open(archivo) as pdf:
                for pagina in pdf.pages:
                    contenido = pagina.extract_text()
                    if contenido:
                        texto_extraido += contenido + "\n"
        except Exception as e:
            st.error(f"Error al extraer texto del PDF: {e}. Asegúrate de que el PDF no esté protegido o sea una imagen.")
            texto_extraido = "" # Asegurarse de que el texto esté vacío en caso de error

    if texto_extraido:
        texto_limpio = limpiar_texto(texto_extraido)

        if texto_limpio:
            if st.button("🔍 Clasificar CV"):
                prob = modelo.predict_proba([texto_limpio])[0]
                pred = modelo.classes_[prob.argmax()]
                max_prob = prob.max()

                # Ajustar el umbral de confianza
                # Puedes experimentar con este valor (ej. 0.5, 0.6, 0.7)
                umbral_confianza = 0.55 
                
                if max_prob >= umbral_confianza:
                    st.success(f"🗂️ Área sugerida para el candidato: **{pred}** (Confianza: {max_prob*100:.2f}%)")
                else:
                    st.warning(f"⚠️ No se pudo clasificar con suficiente confianza ({max_prob*100:.2f}% < {umbral_confianza*100:.2f}%). Por favor revise el CV o considere ajustar el umbral.")

                st.markdown("### 📊 Detalle de Probabilidades")
                # Ordenar las probabilidades de mayor a menor para una mejor visualización
                sorted_probs = sorted(zip(modelo.classes_, prob), key=lambda x: x[1], reverse=True)
                for label, p in sorted_probs:
                    st.write(f"- **{label}**: {p*100:.2f}%")
        else:
            st.warning("⚠️ No se pudo extraer texto significativo del PDF después de la limpieza. Asegúrate de que el archivo contiene texto legible y no es una imagen escaneada.")
    else:
        st.warning("⚠️ No se pudo extraer texto del PDF. El archivo podría estar vacío, protegido o ser una imagen escaneada.")