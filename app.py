import streamlit as st
import pdfplumber
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Entrenamiento del modelo
def entrenar_modelo():
    textos = [
        "Python Java SQL desarrollo web APIs",
        "Diseño gráfico Illustrator Photoshop creatividad",
        "Marketing digital redes sociales campañas",
        "Finanzas contabilidad presupuestos SAP",
        "Machine learning análisis de datos IA"
    ]
    areas = ["Tecnología", "Diseño", "Marketing", "Administración", "Tecnología"]
    modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())
    modelo.fit(textos, areas)
    return modelo

modelo = entrenar_modelo()

st.title("📄 Clasificador de Currículos")
archivo = st.file_uploader("Sube un CV en PDF", type="pdf")

if archivo is not None:
    with pdfplumber.open(archivo) as pdf:
        texto = ""
        for pagina in pdf.pages:
            texto += pagina.extract_text() + "\n"

    if texto.strip() != "":
        if st.button("Clasificar"):
            pred = modelo.predict([texto])[0]
            prob = modelo.predict_proba([texto])[0]
            st.success(f"Área sugerida: **{pred}**")
            st.write("Probabilidades:")
            for label, p in zip(modelo.classes_, prob):
                st.write(f"- {label}: {p*100:.2f}%")
    else:
        st.warning("No se pudo leer el texto del PDF.")
