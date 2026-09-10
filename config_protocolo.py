"""Editar aqui desde VS Code. Las rutas no dependen de la carpeta de la terminal."""
from pathlib import Path

DATA_DIR = Path(r"D:/Doctorado/protocol2023")
OUTPUT_DIR = Path(r"D:/Doctorado/protocol2023/resultados_pipeline")
CACHE_DIR = Path(r"D:/Doctorado/protocol2023/cache_pipeline")
# None = todos los sujetos de config_sujetos.py
SUBJECTS_TO_RUN = ["01_test_2023"]
FEATURES_TO_RUN = ["wpli"]
# FEATURES_TO_RUN = ["spectral", "wpli", "wsmi", "pe", "lzc", "te"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}

STATUS_CHANNEL = "Status"
# Bits de eventos BioSemi; None si no corresponde a otro equipo.
STATUS_MASK = 0xFFFF
# Verificar con los tiempos reales; no desplaza ventanas.
MARK_TOLERANCE_SECONDS = 0.1
WINDOW_SECONDS = 5.0
# "error" o "drop" (descarta solo resto final, lo registra).
REMAINDER_POLICY = "error"
PROTOCOL = {
    40: dict(name="Resting", mode="fixed", start_code=40, duration_s=60.0,
             repeated_offsets_s=[30.0], expected_blocks=1),
    60: dict(name="REY", mode="fixed", start_code=60, duration_s=60.0,
             repeated_offsets_s=[30.0], expected_blocks=1),
    100: dict(name="AUT", mode="fixed", start_code=100, duration_s=60.0,
              repeated_offsets_s=[30.0], expected_blocks=1),
}
# Otros protocolos (reemplazar PROTOCOL, no activar simultaneamente):
# {1: dict(name="Estimulo", mode="fixed", start_code=10, duration_s=20,
#          repeated_offsets_s=[], expected_blocks=None)}
# {1: dict(name="Respuesta", mode="until_marker", start_code=10, end_code=20,
#          min_duration_s=1, max_duration_s=120, expected_blocks=None)}
# {1: dict(name="Evocado", mode="event", start_code=10, tmin_s=-1, tmax_s=3,
#          expected_blocks=None)}
# mode="event": la ventana completa es el trial. fixed/until_marker: subdivide.
# wPLI: una estimacion por bloque experimental con subepocas de WPLI_EPOCH_SECONDS.
WPLI_EPOCH_SECONDS = 5.0
# Control matematico, NO garantia de fiabilidad estadistica.
WPLI_MIN_EPOCHS = 2
EXPORT_CSV = False
REUSE_CLEAN_CACHE = True
PREPROCESS = dict(excluded=["EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"],
                  highpass_cut=1, lowpass_cut=30, interpolate=True)
ICA_SETTINGS = dict(method="infomax", n_components=15, decim=3, random_state=23,
                    reject_limit=250e-6)
SPECTRAL_SETTINGS = dict(freq_range=(1, 30))
WSMI_SETTINGS = dict(embedding_dim=3, tau=None, n_channels_per_zone=None,
                     debug_first_pair=False)
PE_SETTINGS = dict(embedding_dim=3, tau=None, min_samples=200)
# LZC de banda ancha original.
LZC_SETTINGS = dict(freq_range=(1, 30), min_samples=200)
TE_SETTINGS = dict(maxlag_ms=400.0, maxlag=None,
                   min_obs_needed=30, biascorrect=True)
