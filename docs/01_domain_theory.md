# 01 · Teoría de dominio

Qué se detecta, con qué señal y por qué un clasificador con 99 % de accuracy puede ser inútil en un SOC.

## 1. NIDS vs. HIDS

| | NIDS (red) | HIDS (host) |
|---|---|---|
| Punto de observación | TAP / SPAN, firewall, exportador NetFlow | Agente en el endpoint |
| Señal | Paquetes o flujos | Syscalls, procesos, archivos, logs |
| Cobertura | Todo lo que cruza el segmento, incluso dispositivos sin agente (IoT, OT) | Solo hosts instrumentados |
| Ve bien | Escaneos, DoS, fuerza bruta, C2, exfiltración | Escalada de privilegios, persistencia, malware en memoria |
| Puntos ciegos | Payload cifrado, tráfico que no cruza el sensor | Hosts sin agente; un atacante con privilegios puede apagarlo |

Son complementarios. Este proyecto es un NIDS **basado en flujos**: no inspecciona payload, así que sigue funcionando con TLS y escala a enlaces de alto volumen, a costa de no ver el contenido.

Según el método de detección:

- **Por firmas** (Snort, Suricata): preciso con lo conocido, ciego a lo nuevo.
- **Supervisado**: aprende patrones de ataques etiquetados. Generaliza a variantes, no a familias nunca vistas.
- **Por anomalías**: modela lo normal y alerta sobre el desvío. Puede ver lo desconocido, pero *anómalo no implica malicioso*, y de ahí sus falsos positivos.

La arquitectura híbrida combina los dos últimos.

## 2. Flujos bidireccionales

Un flujo agrupa los paquetes que comparten la **5-tupla**:

```
(IP origen, IP destino, puerto origen, puerto destino, protocolo)
```

NetFlow clásico (v5) es *unidireccional*: una conversación TCP son dos registros. CICFlowMeter y Argus generan flujos **bidireccionales**: unifican ambos sentidos y calculan estadísticas por dirección.

- **Forward (fwd):** sentido del primer paquete observado, normalmente cliente → servidor.
- **Backward (bwd):** sentido inverso.

El flujo se cierra por FIN, RST o timeout. Esto tiene consecuencias prácticas:

- Una sesión larga puede partirse en **varios flujos** casi idénticos. Si caen en splits distintos, hay fuga de información.
- Si el primer paquete capturado es del servidor, **la dirección queda invertida** y `Destination Port` pasa a ser el puerto efímero del cliente.
- Las *tasas* (`Flow Bytes/s`) se dividen por la duración: con duración 0 producen `Inf` o `NaN`.

Grupos de features típicos de CICFlowMeter:

| Grupo | Ejemplos | Qué revela |
|---|---|---|
| Volumen | paquetes y bytes fwd/bwd | Asimetría, inundación, exfiltración |
| Tamaño de paquete | min, max, media, desvío | Sondeos vacíos vs. transferencias |
| Tiempos entre llegadas (IAT) | media, desvío, max | Automatización, *beaconing* |
| Flags TCP | SYN, FIN, RST, PSH, ACK | Handshakes incompletos, rechazos |
| Active / Idle | media, max | Conexiones retenidas |
| Ventana inicial TCP | `Init_Win_bytes_*` | Huella de la herramienta o del SO (riesgo de *shortcut*) |

## 3. Firmas de red por familia

| Familia canónica | Etiquetas CIC-IDS2017 | Firma por flujo | Detectable con un solo flujo |
|---|---|---|---|
| `PortScan` | PortScan | 1–2 paquetes, casi sin payload, SYN con respuesta RST o sin respuesta | Sí, aunque la señal fuerte es *agregada* |
| `BruteForce` | FTP-Patator, SSH-Patator | Sesiones cortas y repetidas al mismo servicio, tamaños muy regulares | Parcialmente |
| `DoS` | Hulk, GoldenEye, slowloris, Slowhttptest | Volumétrico o de retención (ver 3.3) | Sí |
| `DDoS` | DDoS | Como el volumétrico, con origen distribuido | Sí |
| `Botnet` | Bot | Conexiones periódicas y pequeñas hacia un C2 | Débil: la periodicidad es entre flujos |
| `WebAttack` | Brute Force, XSS, SQL Injection | HTTP legítimo en forma; el ataque está en el payload | Débil |
| `Infiltration` | Infiltration | Descarga de malware y movimiento lateral | Débil, y muy pocas muestras |
| `Heartbleed` | Heartbleed | Explotación de TLS: respuestas desproporcionadas al pedido. Ocurre el día de DoS, pero no es DoS | Sí, pero hay 11 muestras |

### 3.1 PortScan

Un origen toca muchos puertos o hosts. Según la técnica:

- **SYN scan:** SYN → SYN/ACK (abierto) o RST (cerrado), sin completar el handshake.
- **Connect scan:** handshake completo y cierre inmediato.
- **FIN / NULL / Xmas:** combinaciones de flags inválidas.

Cada flujo aislado es mínimo. La evidencia definitiva, *muchos puertos distintos por origen en una ventana*, requiere agregación.

### 3.2 Fuerza bruta SSH / FTP

Muchas sesiones cortas al 22 o al 21. Cada intento fallido produce un intercambio casi idéntico, por lo que la **varianza entre flujos es bajísima**. SSH está cifrado, pero tamaños e IAT lo delatan. Un intento individual es casi indistinguible de un usuario que erró la contraseña.

### 3.3 DoS volumétrico vs. Slowloris

| | Volumétrico (Hulk, DDoS) | Lento (slowloris, Slowhttptest) |
|---|---|---|
| Recurso agotado | Ancho de banda, CPU | Slots de conexión del servidor |
| Paquetes/s | Muy alto | Muy bajo |
| Duración del flujo | Corta o media | Muy larga |
| IAT | Mínimo | Alto, con *keep-alive* periódico |
| Detección por umbral de volumen | Sí | **No** |

Son firmas opuestas: cualquier regla de volumen que ve al primero ignora al segundo.

### 3.4 C2 beaconing

Un host infectado consulta a su C2 a intervalos regulares, a menudo con *jitter* deliberado, en conexiones salientes pequeñas sobre puertos permitidos (80, 443, 53). Cada flujo parece navegación. La señal está en la **periodicidad entre flujos** del par origen–destino, que las features por flujo no capturan. Es esperable un recall bajo con este esquema.

### 3.5 Ataques web

SQLi y XSS viven en el payload HTTP. A nivel de flujo solo quedan indicios indirectos: muchas peticiones pequeñas y repetidas, o respuestas de error de tamaño constante. Un WAF los ve mejor: es un límite del método, no del modelo.

## 4. Datasets

### 4.1 CIC-IDS2017

Cinco días (3 al 7 de julio de 2017), capturados en un laboratorio con tráfico benigno generado a partir de perfiles de usuario, y flujos con ~80 features de CICFlowMeter. **2 830 743 flujos, 80.3 % benignos.**

| Etiqueta | Flujos | Día |
|---|---:|---|
| BENIGN | 2 273 097 | todos (el lunes es 100 % benigno) |
| DoS Hulk | 231 073 | miércoles |
| PortScan | 158 930 | viernes |
| DDoS | 128 027 | viernes |
| DoS GoldenEye | 10 293 | miércoles |
| FTP-Patator | 7 938 | martes |
| SSH-Patator | 5 897 | martes |
| DoS slowloris | 5 796 | miércoles |
| DoS Slowhttptest | 5 499 | miércoles |
| Bot | 1 966 | viernes |
| Web Attack – Brute Force | 1 507 | jueves |
| Web Attack – XSS | 652 | jueves |
| Infiltration | 36 | jueves |
| Web Attack – SQL Injection | 21 | jueves |
| Heartbleed | 11 | miércoles |

Problemas conocidos que el ETL debe tratar:

- **Dos distribuciones.** `GeneratedLabelledFlows` conserva IPs, puertos y timestamp; `MachineLearningCSV` los quita, salvo `Destination Port`. El split temporal y por sesión **requiere la primera**.
- Encabezados con espacios iniciales y `Fwd Header Length` duplicada.
- `Inf` y `NaN` en `Flow Bytes/s` y `Flow Packets/s`.
- Guion mal codificado en las etiquetas `Web Attack`, y filas vacías en el archivo del jueves a la mañana.
- Timestamps en formato de 12 h sin AM/PM y, en varios archivos, con resolución de minutos.
- Errores de etiquetado y de construcción de flujos documentados en la literatura (Engelen et al., 2021; Lanvin et al., 2022). Los resultados no deben leerse como rendimiento en producción.
- **Cada ataque ocurre en una única ventana**: un corte temporal global deja familias enteras fuera de Train.

### 4.2 UNSW-NB15

Generado en 2015 con IXIA PerfectStorm en el laboratorio de UNSW Canberra; 49 columnas extraídas con Argus, Zeek (Bro) y algoritmos propios. **2 540 044 flujos, 87.4 % benignos.**

| Categoría | Flujos | | Categoría | Flujos |
|---|---:|---|---|---:|
| Normal | 2 218 761 | | Reconnaissance | 13 987 |
| Generic | 215 481 | | Analysis | 2 677 |
| Exploits | 44 525 | | Backdoors | 2 329 |
| Fuzzers | 24 246 | | Shellcode | 1 511 |
| DoS | 16 353 | | Worms | 174 |

- La etiqueta benigna es `Normal` (o vacía en los CSV completos): el adaptador la traduce a `BENIGN`.
- Conserva sus propias categorías como familias: no se fuerzan a la taxonomía de CIC-IDS2017.
- `attack_cat` trae espacios sobrantes y variantes (`Backdoor` / `Backdoors`).
- Las particiones oficiales (175 341 / 82 332) incluyen `id` y no traen IPs ni timestamps: para un split sin fuga se usan los CSV completos, que no tienen encabezado.
- `sttl`, `dttl` y `ct_state_ttl` son casi un proxy de la etiqueta por cómo se generó el tráfico.

### 4.3 Por qué dos

Las taxonomías y las features no coinciden, así que no se mezclan ni se transfiere un modelo de uno al otro. Se valida que **el método**, con las mismas reglas, funcione en ambos.

## 5. El problema real del SOC

### 5.1 Alert fatigue y tasa base

El FPR se multiplica por el volumen benigno. Con 1 000 000 de flujos diarios, 100 de ellos de ataque, y un recall del 90 %:

| FPR | Falsas alertas/día | Verdaderas | Precision |
|---:|---:|---:|---:|
| 1 % | ~10 000 | 90 | 0.9 % |
| 0.5 % | ~5 000 | 90 | 1.8 % |
| **0.1 %** | **~1 000** | **90** | **8.3 %** |

Aun con 0.1 %, 11 de cada 12 alertas son falsas. Por eso el FPR es una **restricción**, no una métrica más: se fija el presupuesto y se maximiza el recall dentro de él. Y por eso la precision medida en un benchmark con 20 % de ataques no es extrapolable a producción.

### 5.2 Desbalance extremo

- **Accuracy** no informa: predecir siempre `BENIGN` da 80 % en CIC-IDS2017.
- **ROC-AUC** es optimista: integra sobre FPR que ningún SOC toleraría.
- **PR-AUC** refleja el costo de los falsos positivos, pero depende de la prevalencia: solo se compara sobre el mismo test.
- El **recall agregado** lo dominan Hulk, PortScan y DDoS (93 % de los ataques). Puede rondar 0.99 con Botnet en 0. De ahí el desglose por familia y el `macro_recall`.
- Con 11 o 21 muestras, un recall de 1.0 no es evidencia: siempre se reporta junto a `n`.

### 5.3 Zero-day

Un clasificador supervisado reparte todo el espacio entre las clases que conoce, y una familia nueva suele caer en `BENIGN`. El Autoencoder aprende solo tráfico normal: lo que reconstruye mal es sospechoso, conocido o no.

La evaluación aleatoria habitual no mide esto, porque todas las familias aparecen en Train. **Leave-One-Attack-Out** lo simula: se quita una familia de Train y Val, y se mide su detección en Test.

Expectativa honesta: el Autoencoder debería detectar desvíos volumétricos o temporales marcados, y tener dificultades con los ataques que imitan tráfico legítimo (WebAttack, Botnet) a 0.1 % de FPR. El protocolo está para cuantificarlo.
