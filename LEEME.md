# Pipeline EEG flexible — primera implementación

Basado en tu última `featureExtraction_1.py`. La copia de referencia está en
`referencia/featureExtraction_original.py`. Esa copia se conserva sin editar.

## Trabajo desde VS Code

Abrí esta carpeta con el intérprete de Python que ya utilizás. Editá
`config_protocolo.py`: las tres rutas absolutas `DATA_DIR`, `OUTPUT_DIR` y
`CACHE_DIR`, y `SUBJECTS_TO_RUN`. Las rutas iniciales apuntan a
`D:/Doctorado/protocol2023`; verificá que correspondan a tu computadora.
En `config_sujetos.py` están los nombres específicos de BDF/FIF sin extensión,
los canales malos y los componentes ICA que excluías en el original.
Si existen BDF y FIF con el mismo nombre, se utiliza BDF y se registra cuál.

Ejecutá `ejecutar_pipeline.py` con **Run Python File**. No requiere argumentos
de terminal ni cambiar de entorno. `features.py` contiene las funciones de
cálculo, pero no es el punto de entrada. Se empieza con el primer sujeto y
`FEATURES_TO_RUN = ["wpli"]`. La línea comentada inmediatamente debajo permite
seleccionar las seis features. No necesita specparam si no seleccionás espectro.

## Organización

| Archivo | Qué se modifica allí |
|---|---|
| `config_protocolo.py` | Rutas, sujetos seleccionados, bandas, marcas, duraciones, selección de features y parámetros |
| `config_sujetos.py` | Canales malos y exclusiones ICA por nombre de archivo |
| `config_regiones.py` | Una única definición de regiones para todos los estimadores |
| `ejecutar_pipeline.py` | Limpieza, caché, selección de intervalos, cálculo y guardado |
| `segmentacion.py` | Reglas de marcas e intervalos, sin alterar el canal Status |
| `features.py` | Estimadores y agregación regional |
| `salidas.py` | Excel, detalles, metadatos y controles de exportación |

## Criterios implementados

- Resting 40, REY 60 y AUT 100: un bloque de 60 s por condición; segunda marca
  del mismo código a los 30 s. La segunda marca verifica el bloque y no inicia
  un segundo bloque independiente. Se esperan dos marcas por condición.
- Se generan doce ventanas consecutivas de 5 s por condición. Los límites son
  `[inicio, fin)`: no se duplica la muestra de unión. Los intervalos se calculan
  después del remuestreo y son relativos al comienzo del Raw.
- **wPLI:** una llamada conjunta por bloque, con doce subépocas de 5 s,
  `MNE-Connectivity`, `method="wpli"`, `mode="fourier"`, `faverage=True`.
  Se mantiene el filtrado IIR por banda del script original. Se refleja el
  triángulo calculado por MNE una sola vez, sin promediar con ceros no calculados.
  La diagonal de canales se deja en cero; la diagonal regional promedia pares
  distintos de canales dentro de esa región. Una región con un solo canal no
  tiene wPLI intrarregional definido: queda NaN y se registra.
- Espectro, wSMI, PE, LZC y TE conservan sus fórmulas y calculan cada ventana.
  Se preservan también sus filtros particulares: compartir la limpieza inicial
  no significa que todos los estimadores tengan el mismo filtrado posterior.
- ICA se ajusta y aplica una vez por sujeto, antes de seleccionar ventanas.
  Las marcas no forman parte de los canales ajustados por ICA. Se guarda el
  Raw limpio y la solución ICA. Se reutilizan si coinciden entrada, huella del
  archivo, parámetros, exclusiones, versiones relevantes y código de limpieza.
- La limpieza conserva notch 50 Hz, referencia EXG1/EXG2, interpolación opcional,
  FIR 1–30 Hz y remuestreo a 500 Hz del original. Para otro equipo, además del
  protocolo hay que revisar estos supuestos y el montaje BioSemi128.
- Bandas: delta 1–4, theta 4–8, alpha 8–12 y beta 12–30 Hz. Se conserva la
  inclusión de ambos extremos que usan los estimadores originales; no se debe
  sumar sin más los resultados de bandas adyacentes como una partición exacta.
- LZC y TE siguen siendo de banda ancha. Se repiten en las filas de las bandas
  para conservar el formato previo; esas filas no son estimaciones distintas
  por banda. Esta convención queda registrada en la configuración de la corrida.

`spec_theta_*` pasa a llamarse `spec_period_*` en la tabla de salida. La cuenta
se mantiene exactamente: integral en la banda de `10**(log10(PSD) - fondo_log10)`,
equivalente a integrar `PSD / fondo_lineal`. No depende de detectar un pico.
El cociente es adimensional y la integral tiene unidades Hz; no es potencia
absoluta en µV² ni la suma de las gaussianas del modelo. Un espectro igual al
fondo produce aproximadamente el ancho integrado de banda, no cero.
Los nombres internos `theta_power_*` se mantienen en los detalles para facilitar
la comparación con tu código; significan esa medida para la banda solicitada.

## Salidas y trazabilidad

Cada ejecución crea una carpeta distinta en `OUTPUT_DIR` y muestra su ruta al
final. No combina silenciosamente resultados nuevos con una tabla anterior.

- `EEG_features_subject_level.xlsx`: hoja `subject_level` con identificadores
  `subject`, `band`, `condition` y las features seleccionadas. Las hojas
  `trazabilidad`, `intervalos`, `incidencias`, `calidad` y `regiones` contienen los
  metadatos separados de las variables destinadas a análisis.
- `detalles/`: resultados por canal, región o par y ventana/bloque, según lo que
  calcula cada estimador. NPZ numérico sin pickle y JSON con nombres, ejes,
  intervalos y parámetros. TE conserva las direcciones y los retardos, en orden
  de 1 a `maxlag_samples`; su diagonal es NaN por definición.
- `calidad`: cantidad de valores finitos, NaN e infinitos por array; también el
  número de ventanas/bloques finitos que contribuyen a cada agregado regional.
  Un promedio con menos contribuciones de las esperadas queda señalado como parcial.
- `eventos/`: marcas originales detectadas, tiempos, intervalos y origen de los
  datos. `regiones` informa canales usados, faltantes e interpolados.
- `configuracion.json`, `codigo/` y `ejecucion.log`: configuración efectiva,
  intérprete, versiones, copia de código y registro de la ejecución.
- CSV opcional con `EXPORT_CSV=True`, generado desde la misma tabla, coma como
  separador y punto decimal. Excel es la salida principal de revisión.

El Excel se reemplaza sólo cuando terminó de escribirse. Si está abierto y
Windows bloquea el guardado, la ejecución se detiene con error; conserva la
versión completa anterior y los detalles ya escritos. No convierte faltantes
en cero. Los cálculos que fallan se registran como fallidos; no se certifica un
resultado por el mero hecho de que el script llegue al final. Revisar el estado
y las incidencias de cada corrida.

## Marcas inesperadas y otros protocolos

Una inconsistencia invalida esa condición, registra el motivo y permite seguir
con las otras condiciones válidas. No acorta un bloque de 60 s para hacerlo
encajar en la grabación. La tolerancia inicial para la marca de 30 s es 0,1 s:
verificarla con los tiempos reales. Esta tolerancia no desplaza las ventanas.
`STATUS_MASK=0xFFFF` selecciona bits de eventos BioSemi; revisarlo al cambiar de
sistema de adquisición.

Hay tres modos configurables, con ejemplos comentados en `config_protocolo.py`:

1. `fixed`: inicio por código y duración fija, con marcas intermedias opcionales.
2. `until_marker`: inicio y final por códigos distintos, con límites de duración.
3. `event`: intervalo relativo a un evento (`tmin_s`, `tmax_s`); ese intervalo
   completo es el trial de las features individuales.

`expected_blocks=None` permite varios bloques. Se calcula wPLI por bloque y se
promedian bloques para la fila agregada, conservando los resultados individuales.
Esto es distinto de estimar wPLI juntando épocas de bloques distintos. El
protocolo actual tiene un único bloque por condición.

Por defecto se rechazan bloques que no se dividen exactamente en ventanas.
`REMAINDER_POLICY="drop"` permite descartar un resto final para las features
por ventana y registra cuántas muestras se descartaron. wPLI requiere división
exacta en sus propias subépocas y al menos `WPLI_MIN_EPOCHS`; dos épocas son un
control mínimo de cálculo, no una garantía de estabilidad. Para trials cortos
hay que definir una estrategia apropiada antes de activar wPLI. Los bloques
solapados se rechazan: no se admiten de forma implícita.

## Verificación incluida y límite de esta entrega

`verificar_pipeline.py` ejecuta pruebas sintéticas de segmentación, wPLI contra
pares solicitados explícitamente a MNE, equivalencia de las otras cinco features
con el original y guardado seguro. `verificar_integracion.py` crea un FIF temporal,
ejecuta las seis features y verifica que una segunda corrida reutilice ICA.
Ambos pueden ejecutarse desde VS Code; no utilizan tus grabaciones.

En esta entrega pasaron cinco grupos de pruebas y la prueba de integración.
La prueba wPLI utiliza doce épocas de 5 s; la integración usa un protocolo corto
sintético para comprobar todos los recorridos sin procesar EEG real. La regresión
numérica compara dos ventanas idénticas entre original y nuevo, con tolerancia
`1e-12` y la misma ubicación de NaN; TE usa retardos hasta 10 ms en esa prueba.
Esto verifica la refactorización en esos casos, no demuestra validez fisiológica
ni estabilidad estadística del estimador.

Entorno probado: Python 3.12.14, MNE 1.6.1, MNE-Connectivity 0.6.0, NumPy 1.26.4,
SciPy 1.13.1, specparam 2.0.0rc3, pandas 2.2.3 y openpyxl 3.1.5. El código usa
sintaxis compatible con Python 3.10; falta ejecutar esta entrega en tu intérprete
3.10 y sobre tus BDF/FIF reales. No actualices tu entorno sólo por esta lista.
La comparación con la referencia requiere las dependencias espectrales incluso
si normalmente elegís sólo wPLI.

Esta entrega abarca extracción. Reshape, normalización, PCA y scripts de R siguen
pendientes de revisar con el nuevo nombre `spec_period_*` y las salidas elegidas.

En la integración sintética, MNE-Connectivity avisó que `EpochsArray` no tenía
anotaciones para incorporar a sus metadatos. Es esperable porque los intervalos
se registran en nuestras tablas y JSON. El aviso se conserva en incidencias;
por ello esa prueba termina como `completed_with_issues` aunque los seis
cálculos y las comprobaciones hayan pasado.
