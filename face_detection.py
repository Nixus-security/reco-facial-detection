"""
Détection et reconnaissance de visages avec système de verrouillage.
- MediaPipe pour la détection rapide
- Tracker maison pour suivre les visages entre les frames (fluidité)
- face_recognition pour l'identification (thread arrière-plan)
- Verrouillage auto si personne inconnue touche clavier/souris
"""

import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import sys
import os
import time
import threading
from pynput import keyboard, mouse
from database import FaceDatabase


# ─── Couleurs ────────────────────────────────────────────────────────────────
COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_REGISTERING = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# ─── Config ──────────────────────────────────────────────────────────────────
TOLERANCE = 0.5
LOCK_WINDOW = "SECURITE - ACCES BLOQUE"
SMOOTH = 0.4  # Lissage des bounding boxes (0 = pas de lissage, 1 = très lissé)


# ═══════════════════════════════════════════════════════════════════════════════
#  FACE TRACKER — suit les visages entre les frames
# ═══════════════════════════════════════════════════════════════════════════════

class TrackedFace:
    """Un visage suivi à travers les frames."""

    def __init__(self, face_id, x, y, w, h):
        self.id = face_id
        # Position lissée
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        # Identité
        self.name = None
        self.color = COLOR_UNKNOWN
        # Durée de vie
        self.frames_since_seen = 0
        self.last_seen = time.time()

    def update_position(self, x, y, w, h):
        """Met à jour la position avec lissage."""
        self.x = self.x * SMOOTH + x * (1 - SMOOTH)
        self.y = self.y * SMOOTH + y * (1 - SMOOTH)
        self.w = self.w * SMOOTH + w * (1 - SMOOTH)
        self.h = self.h * SMOOTH + h * (1 - SMOOTH)
        self.frames_since_seen = 0
        self.last_seen = time.time()

    def get_rect(self):
        return int(self.x), int(self.y), int(self.w), int(self.h)

    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2


class FaceTracker:
    """Associe les détections frame par frame aux visages suivis."""

    def __init__(self, max_distance=100, max_lost_frames=15):
        self.tracked = {}  # id -> TrackedFace
        self.next_id = 0
        self.max_distance = max_distance
        self.max_lost = max_lost_frames

    def update(self, detections):
        """
        Met à jour avec les nouvelles détections [(x, y, w, h), ...].
        Retourne les TrackedFace actifs.
        """
        # Marquer tous comme non vus cette frame
        for face in self.tracked.values():
            face.frames_since_seen += 1

        if not detections:
            self._cleanup()
            return list(self.tracked.values())

        # Calculer les centres des détections
        det_centers = []
        for (x, y, w, h) in detections:
            det_centers.append((x + w / 2, y + h / 2, x, y, w, h))

        # Associer chaque détection au tracked face le plus proche
        used_tracks = set()
        used_dets = set()

        # Construire matrice de distances
        track_ids = list(self.tracked.keys())
        if track_ids:
            for di, (dcx, dcy, dx, dy, dw, dh) in enumerate(det_centers):
                best_tid = None
                best_dist = self.max_distance

                for tid in track_ids:
                    if tid in used_tracks:
                        continue
                    tcx, tcy = self.tracked[tid].center()
                    dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None:
                    self.tracked[best_tid].update_position(dx, dy, dw, dh)
                    used_tracks.add(best_tid)
                    used_dets.add(di)

        # Créer de nouveaux tracked faces pour les détections non associées
        for di, (dcx, dcy, dx, dy, dw, dh) in enumerate(det_centers):
            if di not in used_dets:
                self.tracked[self.next_id] = TrackedFace(self.next_id, dx, dy, dw, dh)
                self.next_id += 1

        self._cleanup()
        return list(self.tracked.values())

    def _cleanup(self):
        """Supprime les visages perdus depuis trop longtemps."""
        to_remove = [
            tid for tid, face in self.tracked.items()
            if face.frames_since_seen > self.max_lost
        ]
        for tid in to_remove:
            del self.tracked[tid]

    def get_active_faces(self):
        """Retourne les visages visibles (vus récemment)."""
        return [
            f for f in self.tracked.values()
            if f.frames_since_seen <= 3
        ]


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD D'IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class IdentificationThread:
    def __init__(self, db):
        self.db = db
        self._frame = None
        self._results = []  # [(center_x, center_y, name, color), ...]
        self._has_known = False
        self._has_unknown = False
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit_frame(self, frame):
        with self._lock:
            self._frame = frame
        self._event.set()

    def get_results(self):
        with self._lock:
            return list(self._results)

    def has_known_face(self):
        with self._lock:
            return self._has_known

    def has_unknown_face(self):
        with self._lock:
            return self._has_unknown

    def _loop(self):
        while self._running:
            self._event.wait(timeout=0.1)
            self._event.clear()

            with self._lock:
                frame = self._frame
                self._frame = None

            if frame is None:
                continue

            try:
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb = np.array(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), dtype=np.uint8)

                locations = face_recognition.face_locations(rgb, model="hog")
                if not locations:
                    with self._lock:
                        self._results = []
                        self._has_known = False
                        self._has_unknown = False
                    continue

                encodings = face_recognition.face_encodings(rgb, locations)
                known_enc = self.db.get_all_encodings()
                known_names = self.db.get_all_names()

                new_results = []
                found_known = False
                found_unknown = False

                for loc, enc in zip(locations, encodings):
                    name = "Inconnu"
                    color = COLOR_UNKNOWN

                    if known_enc:
                        distances = face_recognition.face_distance(known_enc, enc)
                        best = np.argmin(distances)
                        if distances[best] < TOLERANCE:
                            name = known_names[best]
                            color = COLOR_KNOWN
                            found_known = True
                        else:
                            found_unknown = True
                    else:
                        found_unknown = True

                    top, right, bottom, left = loc
                    cx = (left + right) / 2 * 4
                    cy = (top + bottom) / 2 * 4
                    new_results.append((cx, cy, name, color))

                with self._lock:
                    self._results = new_results
                    self._has_known = found_known
                    self._has_unknown = found_unknown

            except Exception:
                pass

    def stop(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  MONITEUR CLAVIER / SOURIS
# ═══════════════════════════════════════════════════════════════════════════════

class InputMonitor:
    def __init__(self):
        self._activity = False
        self._lock = threading.Lock()
        self._enabled = True
        self._last_pos = None
        self._kb = keyboard.Listener(on_press=self._on_input)
        self._ms = mouse.Listener(on_click=self._on_input, on_move=self._on_move)
        self._kb.daemon = True
        self._ms.daemon = True
        self._kb.start()
        self._ms.start()

    def _on_input(self, *args):
        with self._lock:
            if self._enabled:
                self._activity = True

    def _on_move(self, x, y):
        if self._last_pos:
            if abs(x - self._last_pos[0]) > 15 or abs(y - self._last_pos[1]) > 15:
                with self._lock:
                    if self._enabled:
                        self._activity = True
        self._last_pos = (x, y)

    def has_activity(self):
        with self._lock:
            v = self._activity
            self._activity = False
            return v

    def enable(self):
        with self._lock:
            self._enabled = True
            self._activity = False

    def disable(self):
        with self._lock:
            self._enabled = False
            self._activity = False

    def stop(self):
        self._kb.stop()
        self._ms.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS D'AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def assign_identities(tracker, id_results):
    """Associe les résultats d'identification aux visages trackés."""
    for face in tracker.get_active_faces():
        fcx, fcy = face.center()
        best_dist = 120
        best_name = None
        best_color = None

        for (icx, icy, name, color) in id_results:
            dist = ((fcx - icx) ** 2 + (fcy - icy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_name = name
                best_color = color

        if best_name is not None:
            face.name = best_name
            face.color = best_color


def draw_face_box(frame, x, y, w, h, name, color):
    corner_len = min(w, h) // 4
    t = 2

    cv2.line(frame, (x, y), (x + corner_len, y), color, t)
    cv2.line(frame, (x, y), (x, y + corner_len), color, t)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, t)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, t)
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, t)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, t)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, t)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, t)

    if name:
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 14, y), color, -1)
        cv2.putText(frame, name, (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2)


def draw_hud(frame, face_count, known_count, unknown_count, fps, mode, db_size):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    mode_text = "RECONNAISSANCE" if mode == "recognition" else "ENREGISTREMENT"
    mode_color = (100, 255, 100) if mode == "recognition" else COLOR_REGISTERING
    cv2.putText(frame, "FACE DETECTION", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(frame, f"Mode: {mode_text}", (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1)

    fps_color = (0, 255, 0) if fps > 25 else (0, 165, 255) if fps > 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
    cv2.putText(frame, f"DB: {db_size}", (w - 100, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    stats = f"Detectes: {face_count}  |  Connus: {known_count}  |  Inconnus: {unknown_count}"
    cv2.putText(frame, stats, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "[R] Enregistrer  [L] Lister  [D] Supprimer  [P] MdP  [Q] Quitter",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)


def draw_lock_screen(sw, sh, cam_frame, pwd_input, error_msg, identifier):
    screen = np.zeros((sh, sw, 3), dtype=np.uint8)
    cx = sw // 2

    cv2.rectangle(screen, (0, 0), (sw, 100), (0, 0, 60), -1)
    cv2.ellipse(screen, (cx, 45), (20, 16), 0, 180, 360, WHITE, 3)
    cv2.rectangle(screen, (cx - 25, 45), (cx + 25, 80), WHITE, 2)
    cv2.circle(screen, (cx, 62), 4, WHITE, -1)

    title = "ACCES BLOQUE"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.putText(screen, title, (cx - tw // 2, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    sub = "Personne non autorisee detectee"
    (sw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(screen, sub, (cx - sw2 // 2, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)

    cam_h = 180
    cam_w = int(cam_frame.shape[1] * cam_h / cam_frame.shape[0])
    cam_small = cv2.resize(cam_frame, (cam_w, cam_h))
    cam_x = cx - cam_w // 2
    cam_y = 210
    cv2.rectangle(screen, (cam_x - 2, cam_y - 2),
                  (cam_x + cam_w + 2, cam_y + cam_h + 2), WHITE, 1)
    screen[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = cam_small

    if identifier.has_known_face():
        txt, col = "Visage autorise - Deverrouillage...", COLOR_KNOWN
    elif identifier.has_unknown_face():
        txt, col = "Visage non autorise", COLOR_UNKNOWN
    else:
        txt, col = "Aucun visage detecte", (140, 140, 140)
    (rt, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(screen, txt, (cx - rt // 2, cam_y + cam_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    pwd_y = cam_y + cam_h + 55
    cv2.putText(screen, "Mot de passe:", (cx - 140, pwd_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    fy = pwd_y + 10
    cv2.rectangle(screen, (cx - 140, fy), (cx + 140, fy + 38), WHITE, 2)
    stars = "*" * len(pwd_input)
    cv2.putText(screen, stars, (cx - 130, fy + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)
    if int(time.time() * 2) % 2 == 0:
        cur_x = cx - 130 + len(stars) * 16
        if cur_x < cx + 130:
            cv2.line(screen, (cur_x, fy + 6), (cur_x, fy + 32), WHITE, 2)
    cv2.putText(screen, "[ENTREE] Valider", (cx - 55, fy + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    if error_msg:
        (et, _), _ = cv2.getTextSize(error_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(screen, error_msg, (cx - et // 2, fy + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return screen


def draw_registration_prompt(frame, step, name=""):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 4, h // 2 - 50), (3 * w // 4, h // 2 + 50), BLACK, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    if step == "name_entered":
        cv2.putText(frame, f"Enregistrement: {name}", (w // 4 + 15, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_REGISTERING, 2)
        cv2.putText(frame, "Regardez la camera, [ESPACE]", (w // 4 + 15, h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    elif step == "success":
        cv2.putText(frame, f"{name} enregistre !", (w // 4 + 15, h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_KNOWN, 2)
    elif step == "error":
        cv2.putText(frame, "Aucun visage detecte.", (w // 4 + 15, h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_UNKNOWN, 2)


def capture_face_encoding(cap):
    all_enc = []
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        locs = face_recognition.face_locations(rgb, model="hog")
        if locs:
            largest = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
            enc = face_recognition.face_encodings(rgb, [largest])
            if enc:
                all_enc.append(enc[0])
        time.sleep(0.05)
    return np.mean(all_enc, axis=0) if all_enc else None


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    db = FaceDatabase()
    print(f"[INFO] DB: {db.count()} personne(s)")

    if not db.has_password():
        print("\n  ══ CONFIGURATION INITIALE ══")
        while True:
            pwd = input("  Mot de passe (min 4 car.): ").strip()
            if len(pwd) >= 4:
                pwd2 = input("  Confirmer: ").strip()
                if pwd == pwd2:
                    db.set_password(pwd)
                    print("  ✓ Configuré !\n")
                    break
                print("  ✗ Ne correspond pas.\n")
            else:
                print("  ✗ Trop court.\n")

    # ── MediaPipe ──
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    # ── Webcam ──
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Webcam inaccessible.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[INFO] FPS: {cap.get(cv2.CAP_PROP_FPS):.0f}")
    print("=" * 65)
    print("  DÉTECTION + RECONNAISSANCE + VERROUILLAGE")
    print("=" * 65)
    print("  [R] Enregistrer  [L] Lister  [D] Supprimer  [P] MdP  [Q] Quitter")
    print("=" * 65)

    # ── Threads + Tracker ──
    identifier = IdentificationThread(db)
    input_monitor = InputMonitor()
    tracker = FaceTracker(max_distance=100, max_lost_frames=15)

    mode = "recognition"
    reg_name = ""
    reg_step = ""
    reg_timer = 0
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    is_locked = False
    password_input = ""
    lock_error_msg = ""
    lock_error_timer = 0
    unknown_since = 0
    last_activity_time = 0
    LOCK_DELAY = 1.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        fh, fw = frame.shape[:2]

        # ── Détection MediaPipe (chaque frame, ~2ms) ──
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = face_detector.process(rgb_frame)

        raw_detections = []
        if mp_results.detections:
            for det in mp_results.detections:
                bbox = det.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * fw))
                y = max(0, int(bbox.ymin * fh))
                w = min(fw - x, int(bbox.width * fw))
                h = min(fh - y, int(bbox.height * fh))
                if w > 30 and h > 30:
                    raw_detections.append((x, y, w, h))

        # ── Mettre à jour le tracker (lissage des positions) ──
        tracker.update(raw_detections)

        # ── Envoyer au thread d'identification (1 frame sur 3) ──
        if frame_count % 3 == 0:
            identifier.submit_frame(frame)

        # ── Assigner les identités aux visages trackés ──
        id_results = identifier.get_results()
        if id_results:
            assign_identities(tracker, id_results)

        active_faces = tracker.get_active_faces()

        # Déterminer l'état depuis le TRACKER (persistant, pas intermittent)
        tracker_has_known = any(f.name is not None and f.name != "Inconnu" for f in active_faces)
        tracker_has_unknown = any(f.name == "Inconnu" for f in active_faces)
        # Aussi utiliser le thread comme backup
        has_known_thread = identifier.has_known_face()
        has_unknown_thread = identifier.has_unknown_face()
        # Combiner : connu si tracker OU thread dit connu
        has_known = tracker_has_known or has_known_thread
        # Inconnu si tracker OU thread dit inconnu
        has_unknown = tracker_has_unknown or has_unknown_thread

        # ══════════════════════════════════════════════════════════════════
        #  MODE VERROUILLÉ
        # ══════════════════════════════════════════════════════════════════
        if is_locked:
            input_monitor.disable()

            cam_display = frame.copy()
            for face in active_faces:
                if face.name is not None:
                    x, y, w, h = face.get_rect()
                    draw_face_box(cam_display, x, y, w, h, face.name, face.color)

            if lock_error_msg and time.time() - lock_error_timer > 2:
                lock_error_msg = ""

            lock_screen = draw_lock_screen(1920, 1080, cam_display,
                                           password_input, lock_error_msg, identifier)
            cv2.imshow(LOCK_WINDOW, lock_screen)

            if has_known:
                is_locked = False
                password_input = ""
                lock_error_msg = ""
                cv2.destroyWindow(LOCK_WINDOW)
                input_monitor.enable()
                print("  ✓ Déverrouillé (visage) !")
                time.sleep(0.3)
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                if db.check_password(password_input):
                    is_locked = False
                    password_input = ""
                    lock_error_msg = ""
                    cv2.destroyWindow(LOCK_WINDOW)
                    input_monitor.enable()
                    print("  ✓ Déverrouillé (MdP) !")
                    time.sleep(0.3)
                else:
                    lock_error_msg = "Mot de passe incorrect !"
                    lock_error_timer = time.time()
                    password_input = ""
            elif key == 8:
                password_input = password_input[:-1]
            elif key == 27:
                password_input = ""
            elif 32 <= key <= 126:
                password_input += chr(key)
            continue

        # ══════════════════════════════════════════════════════════════════
        #  MODE NORMAL — Logique de verrouillage
        # ══════════════════════════════════════════════════════════════════

        # Accumuler l'activité (ne pas la perdre)
        if input_monitor.has_activity():
            last_activity_time = time.time()

        # Vérifier si verrouillage nécessaire
        # Condition : inconnu présent ET aucun connu ET DB non vide
        should_lock = has_unknown and not has_known and db.count() > 0

        if should_lock:
            if unknown_since == 0:
                unknown_since = time.time()

            # Verrouiller si inconnu depuis LOCK_DELAY ET activité récente (< 2s)
            unknown_duration = time.time() - unknown_since
            recent_activity = (time.time() - last_activity_time) < 2.0

            if unknown_duration > LOCK_DELAY and recent_activity:
                is_locked = True
                password_input = ""
                unknown_since = 0
                print("  ⚠ VERROUILLAGE !")
                cv2.namedWindow(LOCK_WINDOW, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(LOCK_WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                continue
        elif has_known:
            # Reset seulement si un visage connu est confirmé
            unknown_since = 0

        # ── Dessiner les visages trackés ──
        known_count = 0
        unknown_count = 0

        for face in active_faces:
            if face.name is None:
                continue
            x, y, w, h = face.get_rect()
            if face.name == "Inconnu":
                unknown_count += 1
            else:
                known_count += 1
            draw_face_box(frame, x, y, w, h, face.name, face.color)

        face_count = len([f for f in active_faces if f.name is not None])

        # FPS
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_time = now

        draw_hud(frame, face_count, known_count, unknown_count, fps, mode, db.count())

        if mode == "registration" and reg_step:
            draw_registration_prompt(frame, reg_step, reg_name)
            if reg_step in ("success", "error") and time.time() - reg_timer > 2:
                mode = "recognition"
                reg_step = ""

        cv2.imshow("Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("r"), ord("R")) and mode == "recognition":
            mode = "registration"
            reg_name = input("\n  Nom: ").strip()
            if reg_name:
                reg_step = "name_entered"
                print("  → Regardez la caméra, [ESPACE].")
            else:
                print("  → Annulé.")
                mode = "recognition"
        elif key == 32 and mode == "registration" and reg_step == "name_entered":
            print("  → Capture...")
            encoding = capture_face_encoding(cap)
            if encoding is not None:
                db.add_person(reg_name, encoding)
                reg_step = "success"
                print(f"  ✓ {reg_name} enregistré(e) !")
            else:
                reg_step = "error"
                print("  ✗ Échec.")
            reg_timer = time.time()
        elif key in (ord("l"), ord("L")):
            persons = db.list_persons()
            print(f"\n  ── Enregistrés ({len(persons)}) ──")
            for i, n in enumerate(persons, 1):
                print(f"  {i}. {n}")
            if not persons:
                print("  (vide)")
            print()
        elif key in (ord("d"), ord("D")):
            persons = db.list_persons()
            if not persons:
                print("\n  (vide)")
            else:
                for i, n in enumerate(persons, 1):
                    print(f"  {i}. {n}")
                c = input("  Supprimer n° (0=annuler): ").strip()
                try:
                    idx = int(c)
                    if 1 <= idx <= len(persons):
                        db.remove_person(persons[idx - 1])
                        print("  ✓ Supprimé.")
                except (ValueError, IndexError):
                    pass
        elif key in (ord("p"), ord("P")):
            old = input("\n  Ancien MdP: ").strip()
            if db.check_password(old):
                new = input("  Nouveau (min 4): ").strip()
                if len(new) >= 4:
                    new2 = input("  Confirmer: ").strip()
                    if new == new2:
                        db.set_password(new)
                        print("  ✓ Changé !")
                    else:
                        print("  ✗ Ne correspond pas.")
                else:
                    print("  ✗ Trop court.")
            else:
                print("  ✗ Mauvais MdP.")

    face_detector.close()
    identifier.stop()
    input_monitor.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Terminé.")


if __name__ == "__main__":
    main()