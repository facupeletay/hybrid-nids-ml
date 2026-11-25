# ETL y anti-leakage en CIC-IDS2017

Por qué cada paso de `src/etl.py` existe y qué se rompería sin él. Las cifras se midieron sobre los 8 CSV de `GeneratedLabelledFlows` (2 830 743 flujos no vacíos).

## 1. Qué es fuga en un NIDS

Hay fuga cuando el modelo accede, al entrenar, a información que no tendrá en producción o que pertenece a la evaluación. En flujos de red aparece por cuatro vías:

| Vía | Ejemplo | Defensa |
|---|---|---|
| Identidad | La IP del atacante es fija en el laboratorio | Poda de identificadores |
| Tiempo | Cada ataque ocupa una ventana única del día | Poda de `Timestamp`; split cronológico |
| Sesión | Una conexión larga se exporta en varios flujos casi iguales | Purga en la frontera |
| Estadística | Media, varianza o constancia calculadas con Val/Test | Todo ajuste usa solo Train |

## 2. Por qué `MachineLearningCSV` induce fuga

Esa distribución elimina `Flow ID`, IPs, puerto de origen, protocolo y `Timestamp`. Parece más segura, pero quita justamente lo necesario para separar bien:

- **Sin `Timestamp`** no hay orden fiable. El orden de filas no lo sustituye: en martes y miércoles el final del archivo mezcla horas de la mañana y la tarde.
- **Sin 5-tupla** no se sabe qué flujos pertenecen a la misma conexión.

Solo queda un split aleatorio. Con él, los flujos vecinos de una misma ráfaga, o los fragmentos de una misma conexión, quedan a ambos lados, y el modelo se evalúa sobre casi-copias de lo que memorizó.

El ETL exige `GeneratedLabelledFlows`, usa los identificadores **para separar** y los elimina **antes de modelar**. Ante un CSV sin ellos falla con `RawDataError`, en `CicIds2017Adapter.find_sources`.

## 3. Ingesta

| Fuente | Uso |
|---|---|
| UNB (por defecto) | `python -m src.etl --dataset cicids2017` |
| Carpeta propia | `--raw-dir RUTA` |
| Kaggle | `--kaggle-dataset usuario/dataset` |

> **La URL directa de UNB ya no sirve el ZIP** (verificado el 2026-11-25): redirige a una página HTML porque exige completar un formulario de registro. El ETL lo detecta y explica las alternativas, en lugar de guardar HTML con extensión `.zip`. Tras registrarse, alcanza con dejar `GeneratedLabelledFlows.zip` en `data/raw/cicids2017/`.

Los mirrors de Kaggle son de terceros y no oficiales. Tres independientes publican los 8 CSV con idéntico tamaño en bytes; la corrida de referencia usó `hgiabo2k5/generatedlabelledflows`. El SHA-256 de cada CSV queda en la metadata, para comparar contra una copia oficial.

La descarga escribe en un `.part`, reanuda con `Range` y renombra al completar. La extracción usa solo el nombre base de cada entrada, sin confiar en las rutas internas del ZIP.

## 4. Limpieza

Todas las reglas son **por fila o por nombre de columna**: no usan estadísticas, así que no pueden filtrar información entre splits. Por eso se elimina o se acota, y no se imputa.

| Problema | Medido | Política |
|---|---:|---|
| Filas vacías (WebAttacks) | 288 602 | Eliminar |
| Byte `\x96` en `Web Attack` | 2 180 etiquetas | Leer UTF-8 → cp1252 → latin-1 y normalizar |
| `Fwd Header Length` repetida | 8 archivos | Eliminar por nombre |
| Duplicado exacto, metadata incluida | 203 | Eliminar |
| `Inf` / `NaN` en tasas | 2 863 filas | Eliminar |
| Duración o cabecera negativa | 150 filas, todas benignas | Eliminar |
| IAT negativo con duración válida | 2 775 filas | Llevar a 0 |
| `Init_Win_bytes_* = -1` | 35 % y 51 % de las filas | **Conservar** |

**`Inf` y `NaN`.** Coinciden exactamente con `Flow Duration = 0`: la tasa $b/\Delta t$ no está definida. Es el 0.1 % de los flujos, pero incluye 949 de DoS Hulk. Consecuencia operativa: el sistema **no puntúa flujos de duración cero**, y en producción necesitarían una regla aparte.

**IAT negativo.** Son de −1 a −14 µs: paquetes reordenados al capturar, no flujos corruptos. Eliminarlos descartaría 4 de los 11 Heartbleed.

**El centinela −1.** Significa "sin ventana TCP". Una regla ingenua de "negativo = inválido" borraría la mitad del dataset, y sesgada hacia UDP.

**Duplicados de features.** No se eliminan: son tráfico real repetitivo. Se informan (sección 7).

## 5. Reconstrucción del tiempo

Los timestamps son ambiguos: día primero, reloj de **12 horas sin AM/PM** y, salvo el lunes, resolución de **un minuto**.

Las horas presentes son `{8, 9, 10, 11, 12, 1, 2, 3, 4, 5}`, y la captura dura de 08:42 a 17:10. La reconstrucción es entonces biunívoca:

$$
h_{24} = \begin{cases} h + 12 & h < 7 \\ h & h \ge 7 \end{cases}
$$

Sin ella, las 13:00 quedarían antes que las 09:00 y el orden cronológico sería falso. El orden día/mes se resuelve contra la fecha conocida de cada archivo. Un timestamp ilegible, de otra fecha o fuera de 07–19 h detiene el ETL: no se descarta en silencio.

## 6. Split temporal por campaña

### 6.1 Criterio

Una **campaña** es $c = (\text{campaign\_day}, \text{label\_raw})$. Un corte temporal global dejaría familias enteras fuera de Train, porque cada ataque ocurre una sola vez. Se corta dentro de cada campaña. Sea $F_c(t)$ la fracción de flujos con inicio $\le t$, y $t_1 < t_2 < \dots$ sus timestamps distintos:

$$
i^* = \arg\min_i |F_c(t_i) - 0.70|, \qquad
j^* = \arg\min_{j > i^*} |F_c(t_j) - 0.85|
$$

$$
\text{Train}: t \le t_{i^*} \qquad \text{Val}: t_{i^*} < t \le t_{j^*} \qquad \text{Test}: t > t_{j^*}
$$

El corte cae siempre **entre timestamps distintos**, así que el orden es estricto y los empates nunca se reparten. El costo es que las fracciones son aproximadas en campañas breves: PortScan queda 59/28/13, porque un único minuto concentra el 28 % de sus flujos, y GoldenEye, que dura 8 minutos, 73/20/8.

### 6.2 Campañas que van enteras a Test

Con menos de 50 flujos (o menos de tres timestamps), la campaña va **completa a Test**:

| Campaña | Flujos | Motivo adicional |
|---|---:|---|
| Heartbleed | 11 | Es **una sola conexión TCP** partida por el timeout de 120 s: no se puede repartir sin cortarla |
| SQL Injection | 21 | Tres minutos |
| Infiltration | 36 | Seis conexiones |

Son *zero-day* naturales para la capa supervisada. Infiltration no tiene ninguna muestra en Train. El recall sobre ellas es anecdótico y siempre se informa con `n`.

### 6.3 Sesiones en la frontera

Sean $x$ e $y$ flujos de la misma 5-tupla bidireccional, en splits $s(x) < s(y)$. Se purga $x$ si

$$
\max\big(\text{inicio}_y - \text{fin}_x,\ \text{inicio}_x - \text{fin}_y\big) \le g
$$

Se sacrifica siempre el lado anterior: **Test nunca se toca**. La 5-tupla se compara dentro del día y entre campañas, porque un mismo par puede estar etiquetado a veces `BENIGN` y a veces ataque.

**Por pares, no por cadenas.** La definición habitual (encadenar flujos separados por menos de un timeout) falla aquí: DoS Hulk reparte 230 123 flujos en solo 14 108 5-tuplas por reuso de puertos, y encadena toda la campaña en pocas "sesiones". Exigir que cada una quede en un único split purgaba 158 000 flujos de Train.

**Elección de $g$.** El lunes, único día con resolución de 1 s, muestra cuánto tarda una conexión TCP en reaparecer tras el fin del flujo anterior: ≤ 1 s en el 90.6 % de los casos y ≤ 5 s en el 91.5 %. Al resto lo domina el reuso de puertos. Con resolución de un minuto, la separación observada excede a la real en menos de 60 s. De ahí

$$
g = 60 + 5 = 65 \text{ s}
$$

Un $g$ mayor no es gratis, porque la ventana de Val de Hulk dura 3 minutos:

| $g$ (s) | Purgados | Hulk que queda en Val |
|---:|---:|---:|
| 0 | 20 322 | 33 026 |
| **65** | **51 017** | **22 074** |
| 120 | 68 776 | 7 186 |
| 180 | 66 570 | 807 |

Lo que **no** cubre: conexiones ociosas que reaparecen después de 65 s, y la similitud entre conexiones distintas de una misma ráfaga. Esta última es inherente a un dataset con una sola campaña por ataque, y es el motivo de LOAO.

## 7. Resultado de referencia

| | Train | Val | Test |
|---|---:|---:|---:|
| Flujos | 1 928 119 | 427 385 | 421 006 |
| Benignos | 1 576 210 | 337 954 | 340 152 |

Con ~338 000 benignos en Val, un presupuesto de 0.1 % equivale a ~338 falsos positivos: el umbral es estadísticamente estable.

**Gemelos.** Fracción de Test cuyo vector de features ya está en Train:

| PortScan | SSH-Patator | DoS Hulk | FTP-Patator | BENIGN | Bot, DDoS, GoldenEye |
|---:|---:|---:|---:|---:|---:|
| 56 % | 44 % | 28 % | 26 % | 10 % | 0 % |

En PortScan y SSH-Patator, parte del recall será reconocimiento exacto. No es un error de split: son flujos genuinamente idénticos. La evaluación debe informar el recall con y sin gemelos.

## 8. Poda y escalado

**Poda.** Después de asignar el split se eliminan los identificadores (`config.LEAKAGE_COLUMNS`), la metadata interna y las columnas constantes **en Train**: 8, entre flags nunca activados y estadísticas *bulk*. Quedan 70 features. `Destination Port` se conserva, marcada para ablación en `columns.shortcut_prone`.

`campaign_day` viaja como metadata para el análisis y **nunca** es feature: identifica el ataque. Las features se obtienen con `config.get_feature_columns`.

**Escalado.**

| Modelo | Entrada |
|---|---|
| XGBoost / LightGBM | Parquet sin escalar: son invariantes a transformaciones monótonas |
| Autoencoder | `StandardScaler` ajustado solo con `BENIGN` de Train |

Se descartó `RobustScaler`: 18 features tienen rango intercuartílico 0 en el tráfico benigno y quedarían sin escalar, con valores de hasta $1.2 \times 10^8$ (`Idle Max`). `StandardScaler` garantiza varianza unitaria, pero las colas son pesadas ($|z|$ llega a 176 en Val): el Autoencoder debería aplicar además recorte o `log1p`.

## 9. Persistencia

Cada archivo se escribe en un temporal de la misma carpeta y se renombra. La metadata va al final y guarda el SHA-256 de los demás: si el ETL se interrumpe, `load_artifacts` rechaza la mezcla de versiones. El hash del scaler se verifica **antes** de deserializarlo, porque `joblib` ejecuta código.

## 10. Puente al código

| Concepto | `src/etl.py` |
|---|---|
| Ingesta y validación | `CicIds2017Adapter.obtain`, `find_sources`, `download_file` |
| Limpieza | `CicIds2017Adapter.load`, `_clean_numeric` |
| Etiquetas y tiempo | `normalize_label`, `parse_timestamp` |
| Cortes cronológicos | `find_cut_points`, `assign_temporal_split` |
| Purga de sesiones | `find_boundary_flows` |
| Poda y escalado | `drop_leakage_columns`, `find_constant_columns`, `fit_benign_scaler` |
| Gemelos | `count_train_twins` |
| Persistencia | `run_etl`, `load_artifacts` |

Para agregar un dataset alcanza con implementar `DatasetAdapter` y registrarlo en `ADAPTERS`: el split, la poda, el escalado y la persistencia son genéricos.
