# Capa supervisada orientada a SOC

Por qué `src/baseline_xgb.py` mide lo que mide, y qué se concluiría mal sin cada control. Las cifras son de una corrida completa sobre `cicids2017` (XGBoost 3.4.1, CPU, hiperparámetros por defecto, sin búsqueda).

## 1. Protocolo

| Split | Para qué se usa | Para qué no |
|---|---|---|
| Train | Ajustar árboles y `scale_pos_weight` | — |
| Val | Early stopping y umbrales | Ajustar árboles |
| Test | Medir | Cualquier decisión |

Las features entran **sin escalar**: un árbol solo compara contra umbrales, y es invariante a transformaciones monótonas.

`scale_pos_weight` es $n_{\text{benigno}} / n_{\text{ataque}}$ **de Train** (4.48). Desplaza las probabilidades, que dejan de estar calibradas: un score de 0.10 no significa "10 % de probabilidad". No importa para alertar, porque el umbral se calibra después sobre esos mismos scores.

**Val se usa dos veces** (early stopping y umbrales), así que su PR-AUC es algo optimista. Es un sesgo aceptado: las métricas que se informan son las de Test.

## 2. PR-AUC contra ROC-AUC

$$
\text{TPR} = \frac{TP}{P} \qquad \text{FPR} = \frac{FP}{N} \qquad \text{Precisión} = \frac{TP}{TP + FP}
$$

ROC grafica TPR contra FPR, y ninguna depende de la proporción de clases: por eso ROC-AUC no cambia con el desbalance. Ese es justamente su defecto en un SOC. Con $N$ enorme, miles de falsos positivos apenas mueven el FPR, y el área la domina una región (FPR de 10 %, 50 %) en la que nadie operaría.

PR mide la precisión: de cada alerta que abre el analista, cuántas valen. Con prevalencia $\pi$:

$$
\text{Precisión} = \frac{\pi \cdot \text{TPR}}{\pi \cdot \text{TPR} + (1 - \pi) \cdot \text{FPR}}
$$

Con el punto de operación medido (TPR 99.3 %, FPR 0.124 %):

| Prevalencia | Precisión |
|---:|---:|
| 19.2 % (Test) | 99.5 % |
| 1 % | 89.0 % |
| 0.1 % | 44.5 % |
| 0.01 % | 7.4 % |

**La precisión de 99.5 % es una propiedad del laboratorio**, donde uno de cada cinco flujos es ataque. Por lo mismo, el PR-AUC no se compara entre datasets de distinta prevalencia. Recall, FPR y falsas alertas por 100k benignos sí son trasladables.

## 3. Calibración por presupuesto de FPR

El umbral 0.5 no tiene significado operativo. El SOC fija cuántas falsas alertas puede atender, y el umbral es el menor que lo respeta **en los benignos de Val**:

$$
\tau_\alpha = \min \{ \tau : \text{FPR}_{\text{Val}}(\tau) \le \alpha \}, \qquad \alpha \in \{0.1\%, 0.5\%, 1\%\}
$$

Calibrarlo en Test sería elegir el punto de operación conociendo el examen: el FPR informado coincidiría con el presupuesto por construcción y no diría nada sobre tráfico futuro. `calibrate_thresholds` ni siquiera recibe Test, y las pruebas verifican que invertir todas sus etiquetas no cambia el umbral ni un byte del modelo.

**Precisión numérica.** Los scores son `float64`. Con XGBoost se toma el margen y la sigmoide se aplica en `float64`: en `float32` los scores cercanos a 1 se saturan y empatan.

| Presupuesto | Umbral | FPR Val | FPR Test | Recall | Recall sin gemelos | Falsas alertas / 100k |
|---:|---:|---:|---:|---:|---:|---:|
| **0.1 %** | 0.10226 | 0.099 % | **0.124 %** | 99.31 % | 99.05 % | 123.8 |
| 0.5 % | 0.01367 | 0.497 % | 0.486 % | 99.37 % | 99.14 % | 486.3 |
| 1.0 % | 0.00576 | 0.983 % | 0.958 % | 99.42 % | 99.20 % | 958.4 |

PR-AUC en Test 0.99864 (Val: 0.99942); ROC-AUC 0.99955.

**El presupuesto de 0.1 % no se cumple en Test**: 421 falsos positivos contra los 340 admitidos. El exceso ronda las tres desviaciones estándar si los flujos fueran independientes, así que no parece solo azar. Se documenta y no se recalibra. Para operar a 0.1 % real habría que calibrar con margen, y decidirlo en Val.

Pasar de 0.1 % a 1 % multiplica por 7.7 las falsas alertas y gana 0.11 puntos de recall: lo que falta detectar no está cerca del umbral.

## 4. Auditoría de gemelos

Un **gemelo** es un flujo de Test cuyo vector de features existe, idéntico bit a bit, en Train. No es un error del split: PortScan y Patator repiten el mismo paquete miles de veces. Pero sobre ellos el modelo puede **memorizar**, y el recall agregado mezcla memoria con generalización.

La detección es exacta: un hash de 64 bits preselecciona y la igualdad se confirma byte a byte. Se calcula en el espacio de features **del modelo evaluado**, y `-0.0` equivale a `0.0`.

Recall en Test @ FPR 0.1 %:

| Ataque | n | Gemelos | Recall | Recall sin gemelos |
|---|---:|---:|---:|---:|
| PortScan | 21 032 | 56.1 % | 99.82 % | 99.60 % |
| SSH-Patator | 904 | 44.4 % | 99.12 % | 98.41 % |
| DoS Hulk | 33 642 | 28.3 % | 99.61 % | 99.45 % |
| FTP-Patator | 1 234 | 25.5 % | 99.59 % | 99.46 % |
| DoS slowloris | 920 | 3.8 % | 99.35 % | 99.55 % |
| Web Attack - Brute Force | 242 | 3.7 % | 100 % | 100 % |
| DDoS | 20 450 | 0.0 % | 99.87 % | 99.87 % |
| DoS Slowhttptest | 1 204 | 0 | 99.83 % | 99.83 % |
| DoS GoldenEye | 776 | 0 | 100 % | 100 % |
| Web Attack - XSS | 88 | 0 | 97.73 % | 97.73 % |
| **Bot** | 294 | 0 | **0.00 %** | **0.00 %** |

El efecto existe y es chico: 0.22 puntos en PortScan y 0.71 en SSH-Patator. El recall alto no se explica por memorización exacta. En slowloris el signo se invierte: un gemelo no garantiza acierto, porque algunos vectores aparecen en Train con **ambas** etiquetas. En Test hay 926 ataques con un gemelo benigno en Train, y 119 de ellos solo benigno: ningún clasificador puede separarlos con estas features.

**Límite.** Solo detecta copias exactas. Un flujo que difiere en un microsegundo cuenta como nuevo. Es una cota inferior de la similitud entre splits, no una garantía de independencia; la prueba fuerte es LOAO.

## 5. Ablación de `Destination Port`

El puerto es información legítima, pero en el laboratorio cada ataque apunta a un servicio fijo: un modelo puede aprender "22 ⇒ ataque" y parecer un detector. Si al quitarlo el rendimiento colapsa, había aprendido el escenario.

| | Con puerto | Sin puerto | Delta |
|---|---:|---:|---:|
| PR-AUC | 0.99864 | 0.99748 | −0.00116 |
| Recall @ FPR 0.1 % | 99.31 % | 99.15 % | −0.16 pp |
| Recall macro (vistos) | 90.45 % | 90.01 % | −0.44 pp |
| FPR Test | 0.124 % | 0.113 % | −0.011 pp |

**No colapsa.** El puerto concentra el 11 % de la ganancia total, pero es en gran medida sustituible. Quien más lo nota es SSH-Patator (99.12 → 97.46 %).

Dos cuidados al leerla:

- **Los gemelos cambian.** En PortScan pasan de 56 % a 98 %, porque el puerto es casi lo único que distingue un sondeo de otro. Su recall sin puerto (99.97 %) es sobre todo memoria: en los 354 flujos restantes baja a 98.31 %. Las métricas sin gemelos de ambos modelos se miden sobre poblaciones distintas y el reporte las da sin delta.
- **Sin puerto no significa sin atajos.** `min_seg_size_forward` e `Init_Win_bytes_*` son huellas del sistema operativo o de la herramienta, y ganan peso en el modelo ablado. Esta ablación no las evalúa.

## 6. Importancia de features

Por `gain` medio por división, con puerto: `Average Packet Size`, `Bwd Packet Length Std`, `Bwd Header Length`, `Max Packet Length` y `Destination Port`. Las dos primeras reúnen el 64 % de la ganancia total.

El `gain` medio favorece a features usadas en pocas divisiones muy efectivas, y reparte de forma arbitraria entre features correlacionadas. Por eso el reporte incluye también la ganancia total y la cantidad de divisiones. El modelo se recorta a su mejor iteración antes de medir: si no, contarían árboles que no se usan.

LightGBM informa ganancia total y XGBoost media; `top_features` las unifica, pero sus escalas no son comparables entre backends.

## 7. Puntos ciegos

### 7.1 Bot: presente en Train y con recall nulo

Hay 1 378 flujos de Bot en Train y aun así se detectan 0 de 294 en Test, con cualquiera de los tres presupuestos y con o sin puerto. La causa está en los datos:

| | Train | Val | Test |
|---|---:|---:|---:|
| Bot con payload saliente | 1 377 | 87 | 0 |
| Bot sin payload | 1 | 197 | 294 |
| Recall con payload | 99.7 % | 98.9 % | — |
| Recall sin payload | 0 % | 0 % | 0 % |

Los 294 de Test son el mismo patrón: 3 paquetes por sentido hacia el 8080, sin payload, un segundo y `Init_Win_bytes_backward = 0`. La lectura más plausible es un bot que reintenta contra un C2 que **ya no acepta conexiones**: cabeceras de respuesta mínimas (20 bytes), ventana 0 y ningún dato. Es una interpretación, porque los flags registrados no la confirman (`RST Flag Count` vale 0 en todos). Para la red, en todo caso, es una conexión sin contenido.

Un split aleatorio habría repartido ambas fases y mostrado un recall alto. El split cronológico expone que la campaña cambió. Detectarlo requiere contexto entre flujos (*beaconing*), fuera del alcance de un clasificador por flujo.

### 7.2 Ataques sin muestras en Train

Se informan aparte y no entran en `macro_recall_seen`. Recall @ FPR 0.1 %:

| Ataque | n | Con puerto | Sin puerto |
|---|---:|---:|---:|
| SQL Injection | 21 | 21/21 | 18/21 |
| Heartbleed | 11 | 0/11 | 7/11 |
| Infiltration | 36 | 0/36 | 0/36 |

SQL Injection se detecta porque se parece a los otros ataques web, que sí están en Train. Con estos tamaños las diferencias son anecdóticas, y los 11 flujos de Heartbleed son **una sola conexión**: 7/11 no es evidencia de que quitar el puerto generalice. Son el motivo de la capa no supervisada.

### 7.3 LightGBM

Con los mismos hiperparámetros se detiene en 15 árboles: un PR-AUC de Val cercano a 1 deja al early stopping sin gradiente útil. Rinde menos a 0.1 % (recall 98.20 %, slowloris 78.04 %) con FPR de Test 0.101 %. No es una comparación justa entre algoritmos: ninguno se ajustó.

## 8. Artefactos

| Corrida | Sufijo |
|---|---|
| XGBoost | (ninguno) |
| `--model lgbm` | `_lgbm` |
| `--drop-dst-port`, o segunda corrida de `--run-ablation` | `_no_dst_port` |
| `--tag v2` | `_v2`, antes de los anteriores |

El reporte guarda umbrales, features, hash del ETL y SHA-256 del modelo: `load_baseline` rechaza un par que no coincide. Es JSON estricto, y las métricas indefinidas valen `null`.

El modelo se escribe en bytes: en modo texto, Windows convierte `\n` en `\r\n` y LightGBM aborta el proceso al releerlo.

**Device.** `auto` usa CUDA si XGBoost la soporta y `nvidia-smi` lista una GPU. Pedir `cuda` sin GPU es un error, porque XGBoost solo advertiría y entrenaría en CPU. Los wheels de LightGBM no incluyen CUDA y entrenan siempre en CPU. El scoring se hace en CPU.

## 9. Puente al código

| Concepto | `src/baseline_xgb.py` |
|---|---|
| Balance y device | `compute_scale_pos_weight`, `resolve_device` |
| Entrenamiento y early stopping | `train_supervised` |
| Scores `float64` | `score_supervised` |
| Umbrales por presupuesto | `calibrate_thresholds` |
| Gemelos | `find_train_twins`, `summarize_twins` |
| Recall por ataque, vistos y no vistos | `breakdown_by_attack`, `evaluate_test` |
| Ablación | `select_features`, `compare_ablation` |
| Importancia | `top_features` |
| Persistencia | `save_model`, `load_model`, `load_baseline` |
| Orquestación | `fit_and_evaluate` (sin disco), `run` |
