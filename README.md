# Sentiment-Analysis-using-N-grams

Este proyecto implementa modelos de análisis de sentimiento usando **n-gramas** (unigram, bigram y trigram) sobre reseñas de películas del dataset IMDb. Se entrenan tres clasificadores principales: **Naive Bayes, Logistic Regression y SVM**, y se comparan distintos tamaños de n-gramas usando métricas de desempeño.

---

## **Contenido del repositorio**

- `data/` → Dataset IMDb (pequeña versión) o scripts para descargarlo.
- `notebook/` → Jupyter notebooks con la implementación completa.
  - Funciones de limpieza y tokenización.
  - Extracción de features con CountVectorizer / TF-IDF.
  - Entrenamiento y evaluación de modelos.
- `results/` Matrices de confusión, métricas y gráficos de evaluación.

---

## **Metodología**

1. **Preprocesamiento**
   - Convertir a minúsculas, eliminar HTML y caracteres especiales.
   - Tokenizar texto y eliminar stopwords.
   - Mapear etiquetas `positive` → 1, `negative` → 0.
   - División en **train (70%) / val (15%) / test (15%)**.

2. **Extracción de features**
   - Se usan **CountVectorizer + TfidfTransformer** (TF-IDF) o solo conteos.
   - Se prueban **unigramas, bigramas y trigramas** (`ngram_range=(1,1)`, `(1,2)`, `(1,3)`).
   - Se limita el vocabulario a **5000 features**, ignorando palabras muy raras (`min_df=5`) o muy comunes (`max_df=0.8`).

3. **Entrenamiento de modelos**
   - Clasificadores: `MultinomialNB`, `LogisticRegression`, `LinearSVC`.
   - Evaluación en conjunto de validación con:
     - **Accuracy**, **Precision**, **Recall**, **F1-score**.
     - Matrices de confusión visualizadas con `seaborn`.

4. **Evaluación final**
   - Selección del mejor modelo según métricas en validación.
   - Evaluación en conjunto de test para obtener desempeño final.
   - Análisis de errores para identificar casos difíciles (negaciones, sarcasmo, expresiones complejas).

---

## **Resultados**

- Tabla comparativa de métricas por modelo y tamaño de n-gramas.  
- Matrices de confusión y gráficos de precisión/recall/F1.  


