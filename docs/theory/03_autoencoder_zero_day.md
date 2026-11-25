# Autoencoder para ataques desconocidos

Por qué `src/autoencoder.py` puede detectar un ataque que nunca vio, qué tan lejos llega y dónde falla. Las cifras son de una corrida completa sobre `cicids2017` (PyTorch 2.15, CPU, hiperparámetros por defecto, sin búsqueda).

## 1. Hipótesis de variedad

Las 70 features de un flujo no varían con libertad: están atadas por el protocolo y por aritmética (paquetes × tamaño ≈ bytes; duración y paquetes fijan las tasas). El tráfico benigno ocupa entonces una región delgada, aproximable por una variedad $\mathcal{M}$ de dimensión $k \ll 70$.

El encoder $f: \mathbb{R}^{70} \to \mathbb{R}^{12}$ y el decoder $g$ se entrenan **solo con benignos** para que $g(f(x)) \approx x$. Con 12 números no se puede copiar cualquier entrada: la red debe aprender las relaciones que cumple el tráfico normal. Un flujo que las viola no cabe en el cuello de botella y se reconstruye mal.

Dos límites de la hipótesis, que los resultados confirman:

- **Anómalo no implica malicioso.** Un benigno raro también se reconstruye mal, y es la fuente de falsas alertas.
- **Malicioso no implica anómalo.** Un ataque que respeta las relaciones normales se reconstruye bien y es invisible.

## 2. Preprocesamiento

Tres pasos, ajustados **solo con los benignos de Train** y guardados dentro del checkpoint.

### 2.1 `signed_log1p`

$$
\phi(x) = \text{sign}(x) \cdot \ln(1 + |x|)
$$

Es monótona e impar, lineal cerca de 0 y logarítmica en la cola: $10^8 \mapsto 18.4$. Conserva el signo, y con él el centinela `-1` de `Init_Win_bytes_*` ("sin ventana TCP"): $-1 \mapsto -0.69$, distinto de $0 \mapsto 0$. Un `log1p` común daría $-\infty$.

Sin este paso, el MSE queda en manos de las colas: un solo valor a $z = 176$ aporta $176^2 \approx 31\,000$ al error cuadrático, y el gradiente de ese flujo borra el de todos los demás.

### 2.2 Estandarización y recorte

Sobre benignos, tras `signed_log1p`:

| | Filas con algún $\lvert z \rvert > 10$ | $\lvert z \rvert$ máximo en Val |
|---|---:|---:|
| Crudo | 1.43 % | 176 |
| `signed_log1p` | 0.11 % | 106 |

El logaritmo **no elimina** la cola que queda: proviene de flags casi constantes (`Fwd URG Flags`, `RST Flag Count`), cuyo desvío en Train es de 0.007. Cuando valen 1 caen a 106 desvíos, y ninguna transformación monótona de un 0/1 lo evita. El recorte a $[-10, 10]$ (`--clip-z`) acota su aporte al orden de $10^2$. Es determinista y no depende del lote.

Un `NaN` o `Inf` **no** se recorta: se rechaza. Recortarlo lo haría pasar por un valor extremo legítimo.

El `scaler_<dataset>.joblib` del ETL no se usa acá: se ajustó en escala física.

## 3. Arquitectura

```
70 → 64 → 32 → 12 → 32 → 64 → 70        14 482 parámetros
```

Cada capa oculta es `Linear → LayerNorm → SiLU → Dropout(0.1)`.

- **`LayerNorm`, no `BatchNorm1d`.** BatchNorm normaliza con estadísticas del lote: un flujo extremo altera la salida de sus vecinos, y al inferir usa promedios móviles distintos de los del entrenamiento. `LayerNorm` usa solo el propio flujo, así que el score no depende de con quién se lo agrupe, y admite lotes de una fila.
- **`SiLU`** es suave: no tiene unidades muertas.
- **Cuello de botella y salida lineales.** Normalizar o apagar unidades del código latente quitaría capacidad justo donde es escasa, y la salida debe poder tomar cualquier valor estandarizado.

## 4. Entrenamiento sin ataques

| Decisión | Datos que la determinan |
|---|---|
| Media y escala | Train benigno |
| Pesos | Train benigno |
| Época, learning rate | Val benigno |
| Umbral | Val benigno |

`benign_rows` es la única entrada de datos al ajuste, y `train_autoencoder` recibe matrices **sin etiquetas**. Las pruebas verifican que reemplazar o eliminar todos los ataques de Train y Val no cambia un bit de los pesos, del scaler ni del umbral, y que el mismo cambio sobre benignos sí se nota.

`AdamW`, recorte de gradiente a norma 1 y `ReduceLROnPlateau`. El dataset se carga entero en el dispositivo (~0.4 GB) y se baraja con un generador propio, sin `DataLoader`.

La corrida de referencia usó las 50 épocas (la mejor fue la 49): el early stopping no llegó a actuar y la pérdida todavía bajaba despacio.

La pérdida de Train (0.051) supera a la de Val (0.013) por el dropout: en modo evaluación, la de Train es 0.0145. No hay sobreajuste visible.

## 5. Score y umbral

$$
s(x) = \frac{1}{d} \sum_{j=1}^{d} (\hat{x}_j - x_j)^2, \qquad
\tau_\alpha = \min \{ \tau : P_{\text{Val benigno}}(s \ge \tau) \le \alpha \}
$$

`--score-metric` admite también `mae` y `huber`, menos sensibles a una única feature desviada.

La red se evalúa entera en `float64`. No alcanza con convertir al final: en `float32` el score variaría según el tamaño de bloque, y con él qué flujos empatan con el umbral.

El umbral es un **cuantil de los benignos**: no hace falta haber visto un ataque, y es la razón por la que esta capa puede operar contra lo desconocido. Igual que en la capa supervisada, Val benigno se usa dos veces (época y umbral).

## 6. Resultados

ROC-AUC 0.8829, PR-AUC 0.6411.

| Presupuesto | Umbral | FPR Val | FPR Test | Recall | Falsas alertas / 100k |
|---:|---:|---:|---:|---:|---:|
| **0.1 %** | 0.26246 | 0.100 % | **0.141 %** | 9.23 % | 141.1 |
| 0.5 % | 0.14500 | 0.500 % | 0.611 % | 18.85 % | 611.5 |
| 1.0 % | 0.10725 | 1.000 % | 1.163 % | 20.22 % | 1162.7 |

El FPR de Test **excede los tres presupuestos** (480 falsas alertas contra 340 a 0.1 %). El exceso no es uniforme entre capturas: el miércoles llega a 0.261 % y el viernes de PortScan queda en 0.038 %. Se documenta y no se recalibra.

Recall por ataque, ordenado por ROC-AUC contra benignos:

| Ataque | n | ROC-AUC | @ 0.1 % | @ 0.5 % | @ 1 % |
|---|---:|---:|---:|---:|---:|
| Web Attack - Brute Force | 242 | 0.784 | 0 % | 0 % | 0 % |
| Web Attack - XSS | 88 | 0.785 | 0 % | 0 % | 0 % |
| **Web Attack - Sql Injection** | 21 | 0.819 | **0/21** | 0/21 | 0/21 |
| PortScan | 21 032 | 0.819 | 0.02 % | 0.03 % | 0.11 % |
| DoS Hulk | 33 642 | 0.859 | 17.65 % | 32.87 % | 33.27 % |
| SSH-Patator | 904 | 0.859 | 0 % | 0 % | 1.88 % |
| FTP-Patator | 1 234 | 0.950 | 0 % | 0 % | 0 % |
| DoS GoldenEye | 776 | 0.967 | 0 % | 0.52 % | 0.77 % |
| **Infiltration** | 36 | 0.969 | **13/36** | 21/36 | 24/36 |
| DDoS | 20 450 | 0.970 | 5.46 % | 11.43 % | 15.95 % |
| **Bot** | 294 | 0.971 | **0/294** | 0/294 | 0/294 |
| DoS slowloris | 920 | 0.990 | 22.61 % | 68.80 % | 69.35 % |
| DoS Slowhttptest | 1 204 | 0.996 | 14.12 % | 97.18 % | 97.26 % |
| **Heartbleed** | 11 | 0.999 | **9/11** | 11/11 | 11/11 |

### 6.1 Qué aporta

Es un **complemento, no un reemplazo**: su recall global es diez veces menor que el supervisado. Pero ve lo que aquel no:

| Punto ciego | XGBoost @ 0.1 % | Autoencoder @ 0.1 % |
|---|---:|---:|
| Heartbleed | 0/11 | 9/11 |
| Infiltration | 0/36 | 13/36 |
| Bot | 0/294 | 0/294 |

Cada capa consume aquí su propio presupuesto de 0.1 %. Sumadas pueden superarlo: el reparto y el FPR conjunto son tarea de `hybrid_eval`. Los tamaños son mínimos, y los 11 flujos de Heartbleed son **una sola conexión**: es un indicio, no una tasa de detección.

### 6.2 Bot sigue invisible

Su ROC-AUC de 0.971 parece bueno y su recall es nulo. Los 294 flujos son casi idénticos, y sus scores caen en una banda angosta (0.058–0.074) alrededor del percentil **97.0** de los benignos. Hay un 3 % de tráfico normal más raro que ellos, y el presupuesto es de 0.1 %: ningún umbral admisible los alcanza. Es el caso de "malicioso no implica anómalo": una conexión sin contenido es frecuente en tráfico normal. Requiere contexto entre flujos (*beaconing*).

**ROC-AUC engaña a esta escala.** Promedia sobre todos los FPR, y acá solo importa el 1 % izquierdo de la curva.

### 6.3 Quién fija el umbral

En Val, 174 de las 337 falsas alertas a 0.1 % (52 %) son flujos con alguna feature saturada en ±10, y en Test el 54 %. Los flags raros (`RST`, `ECE`, `URG`, `CWE`) aportan un tercio del error en ese grupo.

Son 369 flujos benignos (0.109 % de Val), un conjunto **del tamaño del presupuesto entero**. Es decir: a 0.1 %, la mitad de las alertas se gastan en flags legítimos pero infrecuentes. Bajar `--clip-z` o usar `--score-metric mae` debería liberar presupuesto. No se ajustó: elegirlo mirando el recall de ataques, en Test o en LOAO, convertiría esta capa en supervisada. El criterio admisible es uno sin etiquetas de ataque, como la fracción de alertas benignas de Val que se debe a la saturación.

## 7. Checkpoint

`autoencoder_<dataset><sufijo>.pt` es autocontenido:

| Clave | Contenido |
|---|---|
| `version` | Versión del formato |
| `model_config`, `model_state` | Dimensiones y pesos |
| `preprocessor` | `feature_cols`, `mean`, `scale`, `clip_z` |
| `score_metric` | Métrica con la que se calibró |
| `metadata` | Parámetros, época y hash del ETL |

`.pt` es un pickle, y abrir uno ajeno puede ejecutar código. Se carga con `weights_only=True`, que solo admite tensores y tipos básicos; una prueba confirma que un checkpoint malicioso se rechaza sin ejecutarse. Por eso el preprocesador se guarda como tensores y no como objeto de scikit-learn.

Los umbrales viven en el reporte, junto al SHA-256 del checkpoint: `load_autoencoder` rechaza un par que no coincide. La evaluación usa el checkpoint **releído del disco**: lo que se informa es lo que consumirá `hybrid_eval`.

```python
path, report = load_autoencoder("cicids2017")
scores = score_dataframe(path, df)  # float64, una fila por flujo
```

`score_dataframe` no requiere etiquetas, ignora columnas ajenas y no depende de su orden. Puntuar no exige instalar XGBoost ni LightGBM.

**Reproducibilidad.** Misma semilla, máquina y versión: pesos idénticos bit a bit en CPU. En CUDA se piden kernels deterministas, pero no se probó, y CPU y GPU producen pesos distintos.

## 8. Puente al código

| Concepto | `src/autoencoder.py` |
|---|---|
| Filtro de benignos | `benign_rows` |
| Compresión, escala y recorte | `signed_log1p`, `FlowAEPreprocessor` |
| Red | `FlowAutoencoder` |
| Entrenamiento y early stopping | `train_autoencoder`, `fit` |
| Score `float64` | `reconstruction_error`, `score_dataframe` |
| Umbral | `common.calibrate_thresholds` |
| Evaluación sin umbral y con umbral | `rank_attacks`, `evaluate` |
| Persistencia | `save_checkpoint`, `load_checkpoint`, `load_autoencoder` |
| Orquestación | `run` |
