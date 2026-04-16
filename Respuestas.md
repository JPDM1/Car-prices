# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

---

**Descripción y Análisis:**

El análisis descriptivo realizado en `ejercicio1_descriptivo.py` examina el dataset de precios de vehículos (CarPrice_Assignment.csv) compuesto por 205 instancias y 24 atributos iniciales. El script genera gráficas individuales para cada atributo respecto a la variable objetivo Price, calcula correlaciones, estadísticas descriptivas (media, skewness, curtosis) e identifica atributos con baja correlación que podrían ser excluidos en análisis posteriores.

Se eliminaron las columnas `car_ID` (identificador único sin valor predictivo) y `CarName` (nombre del modelo que no aporta información directa sobre el precio). El dataset resultante contiene 23 atributos, incluyendo variables numéricas (int64, float64) y categóricas (str).

El análisis de correlaciones reveló que 10 atributos numéricos tienen correlación menor a 0.70 con Price, sugiriendo que podrían ser excluidos en análisis futuros para simplificar el modelo sin perder mucha información predictiva.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

**Fuente del dataset:** El dataset proviene del archivo `CarPrice_Assignment.csv` ubicado en la carpeta `data/`. Este dataset es público y está disponible en [Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction).

**Variable objetivo (target):** La variable objetivo es `Price`, que representa el precio del vehículo en dólares.

**Por qué tiene sentido hacer regresión sobre Price:**

1. **Variable continua numérica:** Price es una variable numérica continua, lo cual es apropiado para modelos de regresión que predicen valores numéricos.
2. **Relación con características del vehículo:** El precio de un vehículo depende inherentemente de múltiples características técnicas (tamaño del motor, peso, potencia, eficiencia de combustible, tipo de carrocería, etc.), lo cual sugiere que un modelo de regresión puede capturar estas relaciones.
3. **Interés práctico:** Predecir el precio de vehículos tiene aplicaciones reales en el mercado automotriz para valoración de vehículos usados, análisis de mercado y fijación de precios.
4. **Correlación con múltiples atributos:** El análisis de correlación muestra que varios atributos numéricos tienen correlaciones significativas con Price (algunas >0.70), lo que indica que existe una relación lineal que puede ser modelada con regresión.

---

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

**Distribución de las principales variables numéricas:**

1. **Enginesize (Tamaño del motor):**
   - Distribución: Asimetría positiva (skewness ≈ 1.0), con cola hacia la derecha
   - Media: ≈126.9, Curtosis: ≈1.5
   - Outliers: Se observan algunos valores extremos en el rango superior (motores >300), que corresponden a vehículos de alto rendimiento
   - Decisión: No se eliminaron outliers en este análisis descriptivo

2. **Curbweight (Peso en vacío):**
   - Distribución: Aproximadamente normal con leve asimetría positiva
   - Media: ≈2555.6, Skewness ≈ 0.5, Curtosis ≈ 0.1
   - Outliers: Algunos valores en el rango superior (pesos >4000) correspondientes a vehículos grandes
   - Decisión: No se eliminaron outliers en este análisis descriptivo

3. **Horsepower (Potencia):**
   - Distribución: Asimetría positiva marcada (skewness ≈ 1.0), cola derecha
   - Media: ≈104.3, Curtosis ≈ 0.6
   - Outliers: Valores extremos en el rango superior (>250 hp) correspondientes a vehículos deportivos
   - Decisión: No se eliminaron outliers en este análisis descriptivo

4. **Carwidth (Ancho del vehículo):**
   - Distribución: Aproximadamente normal
   - Media: ≈65.9, Skewness ≈ 0.5
   - Outliers: Mínimos, algunos valores en extremos
   - Decisión: No se eliminaron outliers

**Decisión general sobre outliers:** No se eliminaron outliers en este análisis descriptivo porque:

- Los outliers observados corresponden a vehículos reales (deportivos, de lujo, SUVs) que son parte natural de la distribución del mercado
- Eliminarlos podría sesgar el modelo y perder información sobre segmentos importantes del mercado
- Para un análisis descriptivo, es importante mantener la variabilidad completa del dataset

---

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

**Tres variables numéricas con mayor correlación (en valor absoluto) con Price:**

1. **Enginesize:** Correlación ≈ 0.874
2. **Curbweight:** Correlación ≈ 0.835
3. **Horsepower:** Correlación ≈ 0.808

**Interpretación:**

- Todas las correlaciones son positivas, lo que indica que a mayor tamaño del motor, peso o potencia, mayor es el precio del vehículo
- Enginesize tiene la correlación más alta, lo que sugiere que el tamaño del motor es el predictor más fuerte del precio entre las variables numéricas
- Estas tres variables explican una porción significativa de la variabilidad en el precio, lo cual justifica su uso en modelos de regresión

---

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

**Valores nulos en el dataset:**

- **Resultado:** No hay valores nulos en ninguna de las 23 columnas del dataset
- **Porcentaje:** 0% de valores nulos

**Tratamiento de valores nulos:**

- No fue necesario aplicar ningún tratamiento de valores nulos (imputación, eliminación de filas/columnas) ya que el dataset está completo
- El análisis de `df.isnull().any()` confirmó que todas las columnas tienen valores para todas las 205 instancias
- La ausencia de valores nulos simplifica el preprocesamiento y permite trabajar directamente con el dataset sin necesidad de técnicas de imputación

---

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

**Descripción y Análisis del Preprocesamiento:**

El preprocesamiento para la regresión lineal simple consistió en los siguientes pasos:

1. **Eliminación de columnas irrelevantes**: Se eliminaron `car_ID` (identificador único sin valor predictivo) y `CarName` (nombre del modelo que no aporta información directa sobre el precio).

2. **Selección del atributo predictor**: Se seleccionó `Enginesize` como único predictor para la regresión lineal simple. Esta decisión se basa en el análisis del Ejercicio 1, donde `Enginesize` mostró la correlación más alta con la variable objetivo `Price` (≈0.87), lo que indica una fuerte relación lineal entre el tamaño del motor y el precio del vehículo.

3. **Escalado del predictor**: Se aplicó `StandardScaler` al atributo `Enginesize`. Esta técnica centra los datos en media 0 con desviación estándar 1, lo cual es apropiado para algoritmos de regresión lineal que asumen datos en escalas similares y mejora la estabilidad numérica del modelo.

4. **División Train-Test**: Se dividió el dataset en 80% para entrenamiento (164 muestras) y 20% para prueba (41 muestras) usando `random_state=42` para reproducibilidad.

El preprocesamiento resultó en una matriz X de dimensiones (205, 1) - un solo feature - y un vector y de (205,) con los precios.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

**Valores de métricas en el test set:**

- MAE: 2748.69
- RMSE: 3932.61
- R²: 0.8041

**Coeficientes del modelo:**

- Intercepto (β₀): 13305.12
- Pendiente (β₁): 6889.35
- Ecuación: Price = 13305.12 + 6889.35 × Enginesize

**Evaluación del modelo:**
El modelo funciona razonablemente bien para ser una regresión lineal simple con un solo predictor. El R² de 0.8041 indica que el modelo explica el 80.41% de la variabilidad en el precio usando únicamente el tamaño del motor, lo cual es un resultado sólido considerando la simplicidad del modelo.

No hay evidencia clara de overfitting: las métricas de entrenamiento (R²=0.7507, MAE=2856.77, RMSE=3855.82) y prueba (R²=0.8041, MAE=2748.69, RMSE=3932.61) son similares, con un rendimiento ligeramente mejor en test, lo cual es inusual pero positivo y sugiere que el modelo generaliza bien.

El atributo más influyente es, por definición, `Enginesize` con un coeficiente positivo de 6889.35, lo que indica que por cada unidad adicional de tamaño del motor (en la escala estandarizada), el precio aumenta en promedio $6,889.35. Esto tiene sentido lógico: motores más grandes generalmente implican vehículos más potentes y costosos.

Sin embargo, el MAE de $2,748.69 indica que el modelo tiene un error promedio de casi $3,000 en las predicciones, lo cual es significativo en el contexto de precios de vehículos que van desde $5,000 hasta $45,000. Esto sugiere que aunque `Enginesize` es un predictor importante, el precio de los vehículos depende de muchos otros factores que no están siendo considerados en este modelo simple.

---

**Conclusiones sobre la información útil del Ejercicio 1:**

La información del Ejercicio 1 fue fundamental para el desarrollo de la regresión lineal simple por las siguientes razones:

1. **Selección del predictor**: El análisis de correlación en el Ejercicio 1 identificó que `Enginesize` tenía la correlación más alta con `Price` (≈0.87), lo que justificó su selección como único predictor para la regresión lineal simple. Sin este análisis, habría sido necesario probar múltiples atributos para encontrar el más adecuado.

2. **Validación de la relación lineal**: Los scatter plots del Ejercicio 1 mostraron una relación lineal clara entre `Enginesize` y `Price`, lo que validó que una regresión lineal simple era apropiada para modelar esta relación. Si la relación hubiera sido no lineal, habría sido necesario considerar transformaciones o modelos más complejos.

3. **Identificación de outliers**: Los gráficos del Ejercicio 1 permitieron identificar outliers en `Enginesize` y `Price`. Aunque no se eliminaron en este ejercicio, conocer su presencia ayuda a interpretar por qué el modelo puede tener errores de predicción más altos en ciertos rangos de valores, lo cual se refleja en los residuos.

4. **Justificación del preprocesamiento**: El Ejercicio 1 confirmó que no había valores nulos, lo que simplificó el preprocesamiento. También mostró que `Enginesize` tenía una distribución aproximadamente normal, lo que justificó el uso de StandardScaler para el escalado del predictor.

5. **Interpretación de resultados**: El conocimiento de las estadísticas descriptivas del Ejercicio 1 (media, skewness, curtosis de `Enginesize`) ayuda a interpretar el coeficiente de la regresión. Por ejemplo, saber que `Enginesize` tiene una media de aproximadamente 130 permite contextualizar el impacto del coeficiente de 6889.35 en términos reales del tamaño del motor.

6. **Comparación con regresión múltiple**: El Ejercicio 1 identificó que otros atributos como `Curbweight` y `Horsepower` también tenían correlaciones altas con `Price`. Esto sugiere que una regresión lineal múltiple podría mejorar el R² actual de 0.8041, proporcionando una dirección clara para mejoras futuras del modelo.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

**Descripción y Análisis:**

El Ejercicio 3 implementa una regresión lineal múltiple desde cero utilizando únicamente NumPy, sin emplear scikit-learn para el ajuste del modelo. La implementación utiliza la solución analítica de Mínimos Cuadrados Ordinarios (OLS) para calcular los coeficientes del modelo mediante la fórmula β = (XᵀX)⁻¹ Xᵀy.

El script genera datos sintéticos con semilla fija (seed=42) para garantizar reproducibilidad, con 200 muestras y 3 features. Los coeficientes reales conocidos son β₀=5, β₁=2, β₂=-1, β₃=0.5. Se añade ruido gaussiano (σ=1.5) a la variable objetivo para simular datos reales.

El preprocesamiento incluye:

- División train/test (80%/20%) sin mezcla aleatoria
- Adición de columna de unos a la matriz X para el término independiente (intercepto)
- Cálculo de coeficientes mediante inversión matricial

El modelo se evalúa con métricas MAE, RMSE y R² sobre el conjunto de test, y se genera un gráfico de Valores Reales vs. Valores Predichos para visualizar el rendimiento del modelo.

**Resultados obtenidos:**

- MAE = 1.1665
- RMSE = 1.4612
- R² = 0.6897

El R² de 0.6897 indica que el modelo explica aproximadamente el 69% de la variabilidad en los datos de test, lo cual es aceptable considerando que se usaron datos sintéticos con ruido y que el split no fue aleatorio, lo que puede afectar la representatividad del test set.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

**Explicación de la fórmula β = (XᵀX)⁻¹ Xᵀy:**

Esta fórmula es la solución analítica de Mínimos Cuadrados Ordinarios (OLS) para regresión lineal. Busca encontrar el vector de coeficientes β que minimiza la suma de los errores al cuadrado entre los valores observados y predichos.

**Componentes de la fórmula:**

- **XᵀX**: Es el producto matricial entre la transpuesta de la matriz de features X y X misma. Esto representa la matriz de varianzas-covarianzas de los features.
- **(XᵀX)⁻¹**: Es la inversa de la matriz XᵀX. Esta inversa permite resolver el sistema de ecuaciones lineales.
- **Xᵀy**: Es el producto entre la transpuesta de X y el vector de valores objetivo y. Esto representa la covarianza entre cada feature y la variable objetivo.
- **(XᵀX)⁻¹ Xᵀy**: Al multiplicar la inversa de XᵀX por Xᵀy, se obtiene el vector de coeficientes β que minimiza el error cuadrático medio.

**Por qué es necesario añadir una columna de unos a la matriz X:**

La columna de unos es necesaria para incluir el término independiente o intercepto (β₀) en el modelo. Sin esta columna:

- El modelo solo podría pasar por el origen (0,0), ya que la ecuación sería y = β₁x₁ + β₂x₂ + ... + βₙxₙ
- Con la columna de unos, la ecuación se convierte en y = β₀·1 + β₁x₁ + β₂x₂ + ... + βₙxₙ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- Esto permite que el modelo tenga un "offset" o desplazamiento vertical, lo cual es crucial para ajustar modelos donde la relación entre X e y no pasa necesariamente por el origen
- En términos matricales, añadir la columna de ones hace que el primer coeficiente β₀ se multiplique por 1 para todas las observaciones, proporcionando el intercepto

---

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado | Diferencia |
| --------- | ---------- | -------------- | ---------- |
| β₀        | 5.0        | 4.86499486     | -0.1350    |
| β₁        | 2.0        | 2.06361770     | +0.0636    |
| β₂        | -1.0       | -1.11703839    | -0.1170    |
| β₃        | 0.5        | 0.43851694     | -0.0615    |

**Análisis de la comparación:**
Los coeficientes ajustados están muy cercanos a los valores reales, con diferencias pequeñas (todas menores a 0.15 en valor absoluto). Esto demuestra que la implementación de la solución OLS desde cero funciona correctamente. Las pequeñas discrepancias se deben al ruido gaussiano añadido a los datos (σ=1.5) y al hecho de que el modelo se ajusta solo con el conjunto de entrenamiento (80% de los datos), lo que introduce variabilidad en la estimación de los coeficientes.

---

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

**Valores obtenidos:**

- **MAE** = 1.1665
- **RMSE** = 1.4612
- **R²** = 0.6897

**Comparación con valores de referencia del enunciado:**

- MAE referencia: ≈1.20 (±0.20) → **Obtenido: 1.1665** ✓ (dentro del rango esperado)
- RMSE referencia: ≈1.50 (±0.20) → **Obtenido: 1.4612** ✓ (dentro del rango esperado)
- R² referencia: ≈0.80 (±0.05) → **Obtenido: 0.6897** ✗ (fuera del rango esperado)

**Análisis:**
El MAE y RMSE están dentro de los rangos esperados, lo que indica que el error de predicción es apropiado. Sin embargo, el R² obtenido (0.6897) es menor que el valor de referencia (≈0.80). Esta discrepancia se debe principalmente a que el split train/test no se realiza aleatoriamente (sin mezcla), lo que puede resultar en un test set que no es completamente representativo de la distribución de datos completa. En este caso específico, el test set probablemente contiene observaciones con mayor variabilidad o ruido que reducen el R² calculado.

---

\*_Pregunta 3.4_ — Compara los resultados con la regresión lineal simple del Ejercicio 2 y comprueba si el resultado es parecido. Explica qué ha sucedido.

**Comparación con Ejercicio 2 (Regresión Lineal Simple):**

**Ejercicio 2 (Regresión Lineal Simple con Enginesize):**

- MAE = 2748.69
- RMSE = 3932.61
- R² = 0.8041

**Ejercicio 3 (Regresión Lineal Múltiple con datos sintéticos):**

- MAE = 1.1665
- RMSE = 1.4612
- R² = 0.6897

**Análisis de las diferencias:**

Los resultados **no son comparables** entre sí debido a razones fundamentales:

1. **Diferentes datasets:**
   - Ejercicio 2 usa datos reales de precios de vehículos (dataset CarPrice_Assignment.csv)
   - Ejercicio 3 usa datos sintéticos generados con distribución normal y ruido gaussiano

2. **Diferentes escalas:**
   - Ejercicio 2 trabaja con precios en dólares (rango ≈$5,000-$45,000)
   - Ejercicio 3 trabaja con valores sintéticos en escala estándar (rango mucho menor)

3. **Diferentes objetivos:**
   - Ejercicio 2 es una regresión lineal simple (1 predictor) aplicada a un problema real
   - Ejercicio 3 es una implementación didáctica de regresión lineal múltiple (3 predictors) desde cero con NumPy

4. **Diferentes contextos de evaluación:**
   - Ejercicio 2 usa split train/test aleatorio (random_state=42)
   - Ejercicio 3 usa split train/test sin mezcla (determinístico por orden)

**Conclusión:**
La comparación no tiene sentido porque son ejercicios con propósitos completamente diferentes. El Ejercicio 2 busca resolver un problema real de predicción de precios, mientras que el Ejercicio 3 busca demostrar la comprensión de la implementación matemática de la regresión lineal múltiple usando álgebra lineal.

---

## Ejercicio 4 — Series Temporales

---

Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> _Escribe aquí tu respuesta_

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> _Escribe aquí tu respuesta_

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> _Escribe aquí tu respuesta_

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> _Escribe aquí tu respuesta_

---

_Fin del documento de respuestas_
