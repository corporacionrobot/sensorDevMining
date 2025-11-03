#!/usr/bin/env python3
import sys
import threading
import subprocess
from pathlib import Path
import pygame
import serial, serial.tools.list_ports  # serial

# ── INTENTO cargar OpenCV (para el RTSP). Si no está, usamos FFmpeg.
try:
    import cv2
    CV2_OK = True
except Exception as e:
    cv2 = None
    CV2_OK = False
    print(f"[VIDEO] OpenCV no disponible: {e}")

BASE_DIR = Path(__file__).resolve().parent.parent
RES_DIR  = BASE_DIR / "recursos"

FONDO_PATH = RES_DIR / "Recurso 43@300x.png"
RESIZE_MODE = "fit"

# === SERIAL ===
SERIAL_PORT_GUESS = "/dev/ttyUSB0"   # cambia a /dev/ttyACM0 si tu Nano aparece así
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 0.0
SERIAL_PREFIX = "DIST_MM:"

# === RTSP cámara ===
RTSP_URL = 'rtsp://admin:mining2015@192.168.1.64:554/Streaming/Channels/202'
# El video se ajustará al tamaño/posición del PNG "1@300x.png"
CAM_MATCH_NAME = "1@300x.png"  # nombre del PNG al que igualaremos tamaño y ubicación

# === UMBRALES (metros) para encender 1R desde 6R -> 1R (cercanía) ===
R_THRESHOLDS_M = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]  # ajusta a tu gusto

# === CAPAS DE IMAGEN (siempre visibles) ===
BASE_LAYERS = [
    { "file": RES_DIR / "CAEX@300x.png", "x": 1200, "y": 600, "scale": 1.0 },

    # Serie base "1 -N" (celestes, columna de barras base)
    { "file": RES_DIR / "1 -1@300x.png", "x": 1830, "y": 610, "scale": 1.0 },
    { "file": RES_DIR / "1 -2@300x.png", "x": 1780, "y": 550, "scale": 1.0 },
    { "file": RES_DIR / "1 -3@300x.png", "x": 1730, "y": 490, "scale": 1.0 },
    { "file": RES_DIR / "1 -4@300x.png", "x": 1680, "y": 430, "scale": 1.0 },
    { "file": RES_DIR / "1 -5@300x.png", "x": 1630, "y": 370, "scale": 1.0 },
    { "file": RES_DIR / "1 -6@300x.png", "x": 1580, "y": 315, "scale": 1.0 },
]

# === PNG azules 1..4 (siempre visibles) ===
BLUE_NUM_LAYERS = [
    { "file": RES_DIR / "1@300x.png", "x":  50, "y": 500, "scale": 1.0 },
    { "file": RES_DIR / "2@300x.png", "x": 550, "y": 500, "scale": 1.0 },
    { "file": RES_DIR / "3@300x.png", "x":  50, "y": 900, "scale": 1.0 },
    { "file": RES_DIR / "4@300x.png", "x": 550, "y": 900, "scale": 1.0 },
]
BASE_LAYERS += BLUE_NUM_LAYERS  # se dibujan siempre

# === PNG rojos 1..4 (mismas coords que los azules) — arrancan ocultos ===
RED_NUM_LAYERS = [
    { "file": RES_DIR / "1 RED@300x.png", "x":  50, "y": 500, "scale": 1.0 },
    { "file": RES_DIR / "2 RED@300x.png", "x": 550, "y": 500, "scale": 1.0 },
    { "file": RES_DIR / "3 RED@300x.png", "x":  50, "y": 900, "scale": 1.0 },
    { "file": RES_DIR / "4 RED@300x.png", "x": 550, "y": 900, "scale": 1.0 },
]

# === CAUTION (condicional; arranca oculto) ===
CAUTION_LAYER = { "file": RES_DIR / "CAUTION@300x.png", "x": 1100, "y": 170, "scale": 1.0 }

# === CAPAS "R" controladas por distancia (arrancan ocultas) ===
R_LAYERS = [
    { "file": RES_DIR / "1R -6@300x.png", "x": 1580, "y": 315, "scale": 1.0 },
    { "file": RES_DIR / "1R -5@300x.png", "x": 1630, "y": 370, "scale": 1.0 },
    { "file": RES_DIR / "1R -4@300x.png", "x": 1680, "y": 430, "scale": 1.0 },
    { "file": RES_DIR / "1R -3@300x.png", "x": 1730, "y": 490, "scale": 1.0 },
    { "file": RES_DIR / "1R -2@300x.png", "x": 1780, "y": 550, "scale": 1.0 },
    { "file": RES_DIR / "1R -1@300x.png", "x": 1830, "y": 610, "scale": 1.0 },
]

# === TEXTO ===
TEXT_LAYERS = [
    {
        "text": "--.- M",
        "x": 80, "y": 1500,
        "font_px": 120,
        "color": (0, 255, 255),
        "bold": False,
        "font_path": RES_DIR / "fonts" / "Conthrax-SemiBold.otf",
        "padding": 0,
        "border_radius": 0,
        "align": "left",
    },
]

def ensure_img(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra imagen: {path}")
    return path

def load_image(path: Path) -> pygame.Surface:
    return pygame.image.load(str(path))

def apply_size(img: pygame.Surface, layer: dict) -> pygame.Surface:
    iw, ih = img.get_size()
    w = layer.get("width"); h = layer.get("height"); s = layer.get("scale")
    if w and not h:
        r = w / iw; return pygame.transform.smoothscale(img, (int(w), int(ih * r)))
    if h and not w:
        r = h / ih; return pygame.transform.smoothscale(img, (int(iw * r), int(h)))
    if w and h:
        return pygame.transform.smoothscale(img, (int(w), int(h)))
    if isinstance(s, (int, float)) and s > 0:
        return pygame.transform.smoothscale(img, (int(iw * s), int(ih * s)))
    return img

def compute_scaled_font_px(font_px, s, sx=None, sy=None):
    if RESIZE_MODE == "fit":
        return max(8, int(font_px * s))
    factor = min(sx, sy) if (sx is not None and sy is not None) else 1.0
    return max(8, int(font_px * factor))

def render_text_box(screen, mapper, txt_layer, scale_info):
    map_xy, map_size = mapper
    text = txt_layer.get("text", "")
    color = tuple(txt_layer.get("color", (255, 255, 255)))
    padding = int(txt_layer.get("padding", 0))
    border_radius = int(txt_layer.get("border_radius", 0))
    align = txt_layer.get("align", "left")

    pad_w, pad_h = map_size(padding, padding)
    pad = int((pad_w + pad_h) / 2)
    rr_w, rr_h = map_size(border_radius, border_radius)
    rr = int((rr_w + rr_h) / 2)

    font_px = int(txt_layer.get("font_px", 32))
    s  = scale_info.get("s"); sx = scale_info.get("sx"); sy = scale_info.get("sy")
    scaled_font_px = compute_scaled_font_px(font_px, s, sx, sy)

    font_path = txt_layer.get("font_path")
    if font_path and Path(font_path).exists():
        font = pygame.font.Font(str(font_path), scaled_font_px)
        if txt_layer.get("bold", False): font.set_bold(True)
    else:
        font = pygame.font.SysFont("Consolas", scaled_font_px, bold=bool(txt_layer.get("bold", False))) or pygame.font.Font(None, scaled_font_px)

    text_surf = font.render(text, True, color)

    tx, ty = map_xy(int(txt_layer.get("x", 0)), int(txt_layer.get("y", 0)))
    tw, th = text_surf.get_size()
    box_w, box_h = tw + 2 * pad, th + 2 * pad

    bg = txt_layer.get("bg")
    if bg:
        if len(bg) == 3: bg = (*bg, 255)
        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill((0, 0, 0, 0))
        pygame.draw.rect(box, bg, (0, 0, box_w, box_h), border_radius=rr)
        screen.blit(box, (tx, ty))

    if align == "center":
        text_pos = (tx + (box_w - tw) // 2, ty + (box_h - th) // 2)
    elif align == "right":
        text_pos = (tx + box_w - pad - tw, ty + pad)
    else:
        text_pos = (tx + pad, ty + pad)
    screen.blit(text_surf, text_pos)

# --- Serial helpers ---
def find_serial_port(default_guess: str) -> str:
    if Path(default_guess).exists():
        return default_guess
    ports = [p.device for p in serial.tools.list_ports.comports()]
    for p in ports:
        if "ttyUSB" in p or "ttyACM" in p:
            return p
    return default_guess

def try_open_serial():
    port = find_serial_port(SERIAL_PORT_GUESS)
    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        # 🔽 purga cualquier dato viejo que haya quedado en la cola
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        print(f"[SERIAL] Abierto {port} @ {SERIAL_BAUD}")
        return ser
    except Exception as e:
        print(f"[SERIAL] No pude abrir {port}: {e}")
        return None


def read_dist_mm(ser, leftover: bytearray):
    if ser is None:
        return None
    try:
        # lee exactamente lo disponible (si no hay, intenta un bloque chico)
        avail = getattr(ser, "in_waiting", 0)
        data = ser.read(avail or 256)
        if data:
            leftover.extend(data)

        last_val = None
        # consume TODAS las líneas completas y quédate con la ÚLTIMA válida
        while True:
            nl = leftover.find(b"\n")
            if nl == -1:
                break
            line = leftover[:nl+1]; del leftover[:nl+1]
            try:
                txt = line.decode("utf-8", errors="ignore").strip()
                if txt.startswith(SERIAL_PREFIX):
                    mm_str = txt.split(":", 1)[1].strip()
                    last_val = int(mm_str)
            except Exception:
                pass
        return last_val
    except Exception:
        return None



def build_scaled_surface(layer, map_size):
    img0 = load_image(ensure_img(layer["file"]))
    img0 = apply_size(img0, layer)
    img0 = img0.convert_alpha()
    ow, oh = img0.get_size()
    tw, th = map_size(ow, oh)
    tw = max(1, tw); th = max(1, th)
    return pygame.transform.smoothscale(img0, (tw, th))

# ──────────────────────────────────────────────────────────────────────
#  FFmpeg reader en hilo (funciona aunque no haya OpenCV)
# ──────────────────────────────────────────────────────────────────────
class FFmpegRTSP:
    def __init__(self, url, w, h):
        self.url = url
        self.w = int(w)
        self.h = int(h)
        self.frame_bytes = self.w * self.h * 3  # RGB24
        self.proc = None
        self._latest = None
        self._lock = threading.Lock()
        self._alive = False
        self._thread = None
        self._start()

    @property
    def ok(self):
        return self.proc is not None and self._alive

    def _start(self):
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-vf", f"scale={self.w}:{self.h}",
            "pipe:1",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_bytes*2
            )
        except Exception as e:
            print(f"[VIDEO/FFMPEG] No se pudo iniciar ffmpeg: {e}")
            self.proc = None
            return
        self._alive = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        print("[VIDEO] FFmpeg RTSP iniciado.")

    def _reader(self):
        buf = bytearray()
        try:
            while self._alive and self.proc and self.proc.stdout:
                need = self.frame_bytes - len(buf)
                chunk = self.proc.stdout.read(need)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) == self.frame_bytes:
                    with self._lock:
                        self._latest = bytes(buf)
                    buf.clear()
        except Exception as e:
            print(f"[VIDEO/FFMPEG] Error leyendo frames: {e}")
        finally:
            self._alive = False

    def read(self):
        # Devuelve bytes RGB24 de tamaño exacto (w*h*3) o None si no hay nuevo frame
        with self._lock:
            data = self._latest
            self._latest = None
        return data

    def release(self):
        self._alive = False
        try:
            if self.proc:
                self.proc.kill()
        except Exception:
            pass

def main():
    fondo_path = ensure_img(Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else FONDO_PATH)

    pygame.init()
    pygame.display.set_caption("sensor360 - Overlays + Texto (umbrales + video)")

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    SW, SH = info.current_w, info.current_h
    pygame.mouse.set_visible(False)

    raw_fondo = pygame.image.load(str(fondo_path))
    FW, FH = raw_fondo.get_size()
    fondo_src = raw_fondo.convert_alpha()

    scale_info = {}
    if RESIZE_MODE == "fit":
        s = min(SW / FW, SH / FH)
        DW, DH = max(1, int(FW * s)), max(1, int(FH * s))
        fondo_scaled = pygame.transform.smoothscale(fondo_src, (DW, DH))
        offx, offy = (SW - DW) // 2, (SH - DH) // 2
        def map_xy(x, y):   return (int(offx + x * s), int(offy + y * s))
        def map_size(w, h): return (int(w * s), int(h * s))
        scale_info["s"] = s
    else:
        sx, sy = SW / FW, SH / FH
        fondo_scaled = pygame.transform.smoothscale(fondo_src, (SW, SH))
        offx = offy = 0
        def map_xy(x, y):   return (int(offx + x * sx), int(offy + y * sy))
        def map_size(w, h): return (int(w * sx), int(h * sy))
        scale_info["sx"] = sx; scale_info["sy"] = sy

    # Pre-escalar y posicionar capas base (siempre visibles)
    base_surfs = []
    cam_pos = None
    cam_size = None
    for layer in BASE_LAYERS:
        surf = build_scaled_surface(layer, map_size)
        pos  = map_xy(int(layer.get("x", 0)), int(layer.get("y", 0)))
        base_surfs.append((surf, pos))

        # Si coincide con el PNG que usamos de “marco”, guardamos su rect destino
        try:
            if Path(layer["file"]).name == CAM_MATCH_NAME:
                cam_pos  = pos
                cam_size = surf.get_size()
        except Exception:
            pass

    # Pre-escalar y posicionar PNG rojos 1..4 (arrancan ocultos)
    red_num_surfs = []
    for layer in RED_NUM_LAYERS:
        surf = build_scaled_surface(layer, map_size)
        pos  = map_xy(int(layer.get("x", 0)), int(layer.get("y", 0)))
        red_num_surfs.append((surf, pos))

    # Pre-escalar y posicionar capas R (visibles condicionalmente)
    r_surfs = []
    for layer in R_LAYERS:
        surf = build_scaled_surface(layer, map_size)
        pos  = map_xy(int(layer.get("x", 0)), int(layer.get("y", 0)))
        r_surfs.append((surf, pos))

    # Pre-escalar CAUTION (arranca oculto)
    caution_surf = build_scaled_surface(CAUTION_LAYER, map_size)
    caution_pos  = map_xy(int(CAUTION_LAYER.get("x", 0)), int(CAUTION_LAYER.get("y", 0)))

    # ── Cámara (se ajusta al tamaño exacto de 1@300x.png)
    cap_cv2 = None
    cap_ffm = None
    if cam_pos and cam_size:
        cam_w, cam_h = cam_size
        if CV2_OK:
            cap_cv2 = cv2.VideoCapture(RTSP_URL)
            if not cap_cv2.isOpened():
                print("[VIDEO] No se pudo abrir RTSP con OpenCV. Pasando a FFmpeg …")
                cap_cv2 = None
        if cap_cv2 is None:
            cap_ffm = FFmpegRTSP(RTSP_URL, cam_w, cam_h)
            if not cap_ffm.ok:
                print("[VIDEO] No se pudo abrir RTSP con FFmpeg.")
                cap_ffm = None
    else:
        print("[VIDEO] No encontré la capa base CAM_MATCH_NAME para colocar el video.")

    # Serial / EMA
    ser = try_open_serial()
    leftover = bytearray()
    last_mm = None


    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

        mm = read_dist_mm(ser, leftover)
        if mm is not None:
            last_mm = mm

        # Texto metros SIN EMA en Python (el Arduino ya filtra)
        if last_mm is None or last_mm < 0:
            display_text = "--.- M"
            m = None
        else:
            m = last_mm / 1000.0  # directo, sin suavizado
            display_text = f"{m:0.2f} M"

        if TEXT_LAYERS:
            TEXT_LAYERS[0]["text"] = display_text


        # Cuántos 1R encender (de 0 a 6) según thresholds
        count_on = 0
        if m is not None:
            for thr in R_THRESHOLDS_M:
                if m <= thr:
                    count_on += 1
                else:
                    break
            count_on = max(0, min(count_on, len(r_surfs)))

        # ── DIBUJO
        screen.fill((0, 0, 0))
        screen.blit(fondo_scaled, (offx, offy))

        # ▶️ Video (DEBAJO de overlays 1 y 1 RED): dibujamos ANTES de base_surfs
        if cam_pos and cam_size:
            cam_w, cam_h = cam_size
            if cap_cv2 is not None:
                ret, frame = cap_cv2.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # redimensionar exacto por seguridad
                    frame_res = pygame.image.frombuffer(
                        cv2.resize(frame, (cam_w, cam_h)).tobytes(), (cam_w, cam_h), "RGB"
                    )
                    screen.blit(frame_res, cam_pos)
            elif cap_ffm is not None:
                data = cap_ffm.read()
                if data and len(data) == cam_w * cam_h * 3:
                    surf = pygame.image.frombuffer(data, (cam_w, cam_h), "RGB")
                    screen.blit(surf, cam_pos)

        # base visible siempre (incluye 1@300x.png que hace de marco encima del video)
        for surf, pos in base_surfs:
            screen.blit(surf, pos)

        # encender progresivamente los overlays 1R (6->1)
        for i in range(count_on):
            surf, pos = r_surfs[i]
            screen.blit(surf, pos)

        # ⚠️ Mostrar CAUTION al pasar la tercera franja
        if count_on >= 3:
            screen.blit(caution_surf, caution_pos)

        # Si hay al menos un overlay 1R, mostrar “1 RED” por encima
        if count_on >= 1 and len(red_num_surfs) >= 1:
            red1_surf, red1_pos = red_num_surfs[0]  # 1 RED
            screen.blit(red1_surf, red1_pos)

        # textos
        mapper = (lambda x,y: (offx + int(x*(scale_info.get("s",1) if "s" in scale_info else scale_info["sx"])),
                               offy + int(y*(scale_info.get("s",1) if "s" in scale_info else scale_info["sy"]))),
                  lambda w,h: (int(w*(scale_info.get("s",1) if "s" in scale_info else scale_info["sx"])),
                               int(h*(scale_info.get("s",1) if "s" in scale_info else scale_info["sy"]))))
        for txt in TEXT_LAYERS:
            render_text_box(screen, mapper, txt, scale_info)

        pygame.display.flip()
        clock.tick(30)

    # Cierre ordenado
    try:
        if ser: ser.close()
    except Exception:
        pass
    try:
        if cap_cv2 is not None:
            cap_cv2.release()
        if cap_ffm is not None:
            cap_ffm.release()
    except Exception:
        pass
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        try: pygame.quit()
        except Exception: pass
        raise
