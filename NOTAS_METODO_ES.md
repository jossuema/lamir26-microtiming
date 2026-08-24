# Microtiming as a Genre Fingerprint — LAMIR 2026

Código del paper *"Is Microtiming a Genre Fingerprint? Timing Discriminability in
Candombe and Samba Drumming"* (envío a LAMIR 2026). Ver [abstract.md](abstract.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install mirdata librosa numpy pandas scikit-learn matplotlib soundfile pyarrow
# el Python de python.org en macOS necesita el bundle de certifi para descargar:
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")
```

Descarga de datos (una vez, ~1.5 GB):

```python
import mirdata
for name in ["candombe", "brid"]:
    mirdata.initialize(name, data_home=f"./data/{name}").download(cleanup=True)
```

## Uso

```bash
.venv/bin/python scripts/00_synthetic_test.py      # validación sin audio (10 checks)
.venv/bin/python scripts/01_extract_deviations.py  # corpus -> results/*.parquet
.venv/bin/python scripts/02_sanity_check.py        # perfiles + checks vs literatura
```

## Método (resumen)

- Grilla: K=4 subdivisiones isócronas por intervalo entre beats ANOTADOS
  (candombe: 35 grabaciones; BRID: las 93 mezclas con beats — los solos no tienen).
- Onsets: flujo espectral por banda de registro (hop=64 → 2.9 ms/frame,
  interpolación parabólica sub-frame), gate de dominancia entre bandas contra
  el leakage de ataques broadband.
- Desviación = (t_onset − t_grid)/duración_beat; celda de Voronoi completa
  (|dev| ≤ 0.125) para no censurar la cola de anticipación.
- Atribución por rol: bandas de registro (chico/repique/piano;
  surdo/caixa/tamborim). **Pendiente día 2:** calibrar latencia por banda en
  pistas solistas de BRID antes de comparar ENTRE roles.

## Decisiones metodológicas del día 2 (calibración, tras 2.ª revisión adversarial)

- **El gate de dominancia v1 estaba roto por construcción** (verificado
  empíricamente: una banda 58 dB por debajo pasaba el 96 % de las veces). Las
  envolventes de flujo en dB son casi invariantes al nivel, y la normalización
  por percentil propio borra lo que queda. El gate v2 compara **potencia mel
  media por fila en dB** (escala común): un pico en la banda b sobrevive solo
  si su potencia está a ≤ `gate_db` del máximo de las otras bandas
  (default 20 dB — el leakage real está decenas de dB abajo).
- **El ancla de calibración v1 era inválida**: en un solo, el pico full-band ES
  el propio instrumento, así que los "diferenciales" banda-vs-full eran
  degenerados. v2 ancla cada onset de banda al **inicio físico del ataque**
  (RMS causal — sin smearing hacia atrás ni pre-ringing de Hilbert — validado
  a <1 ms en pulsos sintéticos de 3 kHz y 60 Hz en `00_synthetic_test`).
- **Re-alcance de la calibración**: el centrado por extracto anula offsets
  constantes por banda → los claims cross-genre (centrados + like-register)
  son invariantes a latencia y NO necesitan corrección. La calibración sirve
  para claims cross-ROL en crudo dentro de BRID; transferir a candombe exige
  el criterio de equivalencia (spread entre instrumentos ≤ 2 ms por banda) Y
  validación en pseudo-mezclas.
- **La matriz de atribución sobre solos es solo diagnóstico**: la confusión
  real se mide en pseudo-mezclas (suma de solos surdo+caixa+tamborim del mismo
  ritmo, ground truth = ataques de los solos aislados). Los solos no están
  alineados en fase → la precisión en posiciones coincidentes es una cota
  optimista; se reporta.
- **Atribución a nivel de onset individual es poco confiable en mezclas
  densas** (medido): el paper habla de **bandas de registro** (grave/media/
  aguda), no de instrumentos, y reporta la incertidumbre de atribución medida.

### Resultados empíricos del día 2 (con el detector corregido)

- **Latencia absoluta del detector ≈ 30 ms** (pico de flujo vs inicio físico
  del ataque), consistente entre solos (28–34 ms en bandas media/aguda) y
  **validada en pseudo-mezclas (29.2–30.9 ms, desacuerdo ≤ 2 ms)**. Explica el
  offset crudo de +4–8 % del beat del día 1.
- **Criterio de transferencia: FALLA** — la latencia es instrument-driven
  (spread entre instrumentos por banda: 10–30 ms ≫ margen de 2 ms). No hay
  corrección transferible a candombe; los claims cross-genre quedan en
  perfiles centrados + like-register (invariantes a latencia constante).
  La tasa de offsets implausibles (22–66 %) en la medición sobre solos indica
  además que en ejecución densa el estimador de ataque ancla a veces al evento
  vecino: los números absolutos son escala, no corrección de precisión.
- **Punto de operación del gate: `gate_db=20`** (sweep None/30/20/10 en
  pseudo-mezclas): a 10 dB el recall del tamborim colapsa (0.21) sin ganancia
  de precisión; a 20 dB se rechaza el leakage unilateral >20 dB con costo
  moderado. Incertidumbre de atribución medida en el punto de operación:
  precisión own-instrument 0.50–0.70, recall 0.45–0.82, con coincidencias
  ambiguas frecuentes (las partes se entrelazan en la subdivisión) — por eso
  los perfiles se interpretan a nivel de REGISTRO, no de instrumento.
- **Diagnóstico de solos con gate corregido** (leakage bajo auto-normalización
  eliminado): surdo solo → 91 % a su banda; tantã → 69 % grave; el tamborim
  reparte 47/32 entre media y aguda (su energía cruza el borde de 1500 Hz).

## Decisiones del día 6 (clasificador, tras 4.ª revisión adversarial)

- **Features = los 6 contrastes identificables** (cliff/tail × low/mid/high);
  las 12 posiciones centradas del v1 tenían rango 8 y las importancias las
  repartía la penalización L2, no los datos. Banda "full" fuera (colineal).
- **Muestra primaria = caso completo (n=103)**: las 22 pistas con celdas
  imputadas eran TODAS de samba (la ausencia era pista de género). Acotado:
  clasificador de solo-ausencias ≈ azar (0.617); sin registro agudo, 0.813.
- **Métricas descriptivas en leave-one-pair-out** (24 folds con ambas
  clases): el AUC agrupado sobre folds de una sola clase era inválido (y
  revertía el ranking timing-vs-BPM). Inferencia = permutación exacta de
  sesiones sobre bal_acc LOGO agrupada (inmune al artefacto).
- **Resultados:** timing solo 0.823/0.826, p exacto = 0.024. BPM solo 0.891
  (p=0.006) — más fuerte que timing, se reporta sin vueltas. Combinado 0.926,
  pero **Δ complementariedad = +0.022, p=0.291 (ns)**: la evidencia de que el
  timing agrega más allá del tempo es la ANCOVA del día 5, no el clasificador.
- **Identidad multinivel (hallazgo):** 3 particiones intra-candombe superan al
  etiquetado real (hasta 0.881 vs 0.826); discriminación por pares de
  sesiones: candombe 0.964, samba 0.581. El microtiming codifica identidad de
  ensamble al menos tan fuerte como identidad de tradición — enmarcar como
  resultado (consistente con σ_sesión del piano y con los clusters por
  intérprete de Fuentes et al. 2019).
- **Importancias:** coeficientes líderes low_cliff (−1.38, la V del piano) y
  high_tail (+1.40); la ablación muestra redundancia entre registros (solo
  low es individualmente necesario, +0.08). Reportar ambas vistas.
- **Redacción del claim:** "los CORPUS son discriminables por la forma del
  timing solo" — género ⟂ corpus; atribución de género condicional, con
  párrafo de explicaciones alternativas (rutas de medición/anotación).

## Decisiones metodológicas registradas (de la 1.ª revisión adversarial)

1. **Offset positivo uniforme** en los perfiles (~+2 a +8 % del beat) = latencia
   del detector (pico de flujo tras el ataque). El análisis usa la FORMA del
   perfil; centrar por extracto elimina el componente constante y la latencia
   dependiente de tempo.
2. **Posición 1 es parcialmente circular** (la grilla ancla en el beat anotado,
   y las anotaciones suelen caer sobre onsets tocados): las features del
   clasificador y los claims cross-genre usan posiciones 2–4.
3. **CIs entre pistas, no entre onsets** (pseudoreplicación): agregación
   per-track primero, luego CI t-based con n = grabaciones.
4. **Unidades de agrupamiento para CV sin leakage:** candombe → 8 grupos de
   ensamble (`csic.1995_ansina1`, …); BRID → set de músicos (`M2`–`M4`, 3
   sets). Leave-one-GROUP-out, no leave-one-recording-out puro.

## Decisiones del día 5 (estadística, tras 3.ª revisión adversarial)

- **Centrado sobre posiciones 2–4 solamente** (centrar sobre 1–4 filtraba la
  convención de anotación de la pos. 1, confundida con el corpus, como −Δ/4).
- **Inferencia primaria = ANCOVA por permutación exacta de sesiones**
  (etiquetas de género permutadas sobre las C(11,3)=165 asignaciones posibles,
  coeficiente de género con BPM como covariable) + Holm sobre la familia de 6
  contrastes ("cliff" = p2−(p3+p4)/2; "tail" = p3−p4; 2 gl del perfil
  centrado). El bootstrap de 3 clusters del v1 era inválido (~10 remuestreos
  distintos) y el test crudo de sesión no sobrevive Holm (potencia baja con
  11 grupos): se reporta como chequeo conservador.
- **Resultado central: 4/6 contrastes de género sobreviven la ANCOVA
  tempo-controlada con Holm** (low-tail, mid-cliff, high-cliff, high-tail,
  p=0.036) — las diferencias de género en la forma del timing no se reducen a
  tempo. En varios, BPM actúa como supresor (g crudo ≈ 0 pero coeficiente
  ANCOVA grande): graficar el coeficiente, no el g crudo, en el paper.
- **Subgéneros: sin efecto demostrable más allá del tempo** (test limpio
  samba vs partido alto con BPM covariable y permutación intra-sesión: todo
  ns; el enredo difiere descriptivamente pero está separado por tempo). El
  subset por rango de BPM queda como sensibilidad etiquetada (cambia la
  composición: enredo 23 %→51 %).
- **Dependencia de tempo específica por registro** (chequeo de unidades en
  ms): real en caixa (ρ_ms=0.50), mayormente artefacto de fracción-de-beat en
  tamborim, débil en surdo.
- **Varianza (MoM):** en samba σ_sesión ≪ σ_pista (0.04–0.27 vs 0.9–1.5); en
  candombe el piano invierte (σ_sesión=1.07 > σ_pista=0.64): el tambor piano
  carga la identidad del ensamble. Justifica el CV agrupado.
- **Limitación estructural para el paper:** género ⟂ corpus (un corpus por
  género); todos los claims son "candombe/samba según estos corpus". El día 6
  debe incluir baseline de solo-BPM y control de discriminabilidad de
  sesiones intra-corpus.
5. **BRID no es todo samba:** las mezclas anotadas incluyen marcha (3 pistas);
   los perfiles de samba filtran `genre == "samba"` (SA/SE/PA) y se reporta lo
   excluido.

## Estructura

```
src/microtiming/   grid.py · onsets.py · deviations.py · pipeline.py
scripts/           00_synthetic_test · 01_extract_deviations · 02_sanity_check
data/              candombe/ · brid/   (no versionar)
results/           parquets + figuras  (regenerables)
```
