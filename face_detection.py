"""
Détection et reconnaissance de visages en temps réel via webcam
- Haar Cascade pour la détection rapide (temps réel)
- face_recognition pour l'identification (thread arrière-plan)
- Zéro lag sur l'affichage caméra
"""

import cv2
import numpy as np
import face_recognition
import sys
import os
import time
import threading
from database import FaceDatabase


# ─── Couleurs ────────────────────────────────────────────────────────────────
COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_PENDING = (200, 200, 200)
COLOR_REGISTERING = (0, 165, 255)
WHITE = (255, 255, 255)

# ─── Config ──────────────────────────────────────────────────────────────────
TOLERANCE = 0.5


class IdentificationThread:
    """
    Thread d'identification en arrière-plan.
    Reçoit une frame, encode les visages, les compare à la DB.
    Le thread principal ne bloque jamais.
    """

    def __init__(self, db):
        self.db = db
        self._frame = None
        self._identities = {}  # { (top,right,bottom,left) : (name, color) }
        self._lock = threading.Lock()
        self._has_new_frame = threading.Event()
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit_frame(self, frame):
        """Envoie une frame pour identification (non bloquant, drop si occupé)."""
        with self._lock:
            self._frame = frame
        self._has_new_frame.set()

    def get_identities(self):
        """Récupère le dernier résultat d'identification."""
        with self._lock:
            return dict(self._identities)

    def _loop(self):
        while self._running:
            # Attendre une nouvelle frame (timeout pour pouvoir quitter)
            self._has_new_frame.wait(timeout=0.1)
            self._has_new_frame.clear()

            with self._lock:
                frame = self._frame
                self._frame = None

            if frame is None:
                continue

            try:
                # Réduire pour accélérer l'encodage
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb = small[:, :, ::-1].copy()

                locations = face_recognition.face_locations(rgb, model="hog")

                if len(locations) == 0:
                    with self._lock:
                        self._identities = {}
                    continue

                encodings = face_recognition.face_encodings(rgb, locations)

                known_encodings = self.db.get_all_encodings()
                known_names = self.db.get_all_names()

                new_ids = {}
                for loc, enc in zip(locations, encodings):
                    name = "Inconnu"
                    color = COLOR_UNKNOWN

                    if len(known_encodings) > 0:
                        distances = face_recognition.face_distance(known_encodings, enc)
                        best = np.argmin(distances)
                        if distances[best] < TOLERANCE:
                            name = known_names[best]
                            color = COLOR_KNOWN

                    # Stocker en coordonnées originales
                    top, right, bottom, left = loc
                    new_ids[(top * 4, right * 4, bottom * 4, left * 4)] = (name, color)

                with self._lock:
                    self._identities = new_ids

            except Exception as e:
                pass  # Ne jamais crasher le thread

    def stop(self):
        self._running = False
        self._has_new_frame.set()
        self._thread.join(timeout=2)


def match_face_to_identity(face_rect, identities, threshold=80):
    """
    Associe un rectangle Haar Cascade à une identité connue du thread.
    Utilise la distance entre les centres des rectangles.
    """
    fx, fy, fw, fh = face_rect
    face_cx = fx + fw // 2
    face_cy = fy + fh // 2

    best_name = None
    best_color = COLOR_PENDING
    best_dist = threshold

    for (top, right, bottom, left), (name, color) in identities.items():
        id_cx = (left + right) // 2
        id_cy = (top + bottom) // 2
        dist = abs(face_cx - id_cx) + abs(face_cy - id_cy)

        if dist < best_dist:
            best_dist = dist
            best_name = name
            best_color = color

    return best_name, best_color


def draw_face_box(frame, x, y, w, h, name, color):
    """Dessine un cadre stylisé autour du visage avec le nom."""
    corner_len = min(w, h) // 4
    thickness = 2

    cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    if name:
        label = name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 14, y), color, -1)
        cv2.putText(frame, label, (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def draw_hud(frame, face_count, known_count, unknown_count, fps, mode, db_size):
    """Dessine le HUD."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    mode_text = "RECONNAISSANCE" if mode == "recognition" else "ENREGISTREMENT"
    mode_color = (100, 255, 100) if mode == "recognition" else COLOR_REGISTERING
    cv2.putText(frame, "FACE DETECTION", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)
    cv2.putText(frame, f"Mode: {mode_text}", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    cv2.putText(frame, f"DB: {db_size}", (w - 110, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    stats = f"Detectes: {face_count}  |  Connus: {known_count}  |  Inconnus: {unknown_count}"
    cv2.putText(frame, stats, (15, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    ctrl = "[R] Enregistrer  [L] Lister  [D] Supprimer  [Q] Quitter"
    cv2.putText(frame, ctrl, (10, h - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return frame


def draw_registration_prompt(frame, step, name=""):
    """Affiche les instructions d'enregistrement."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 4, h // 2 - 60), (3 * w // 4, h // 2 + 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    if step == "name_entered":
        cv2.putText(frame, f"Enregistrement de: {name}", (w // 4 + 20, h // 2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_REGISTERING, 2)
        cv2.putText(frame, "Regardez la camera, appuyez [ESPACE]", (w // 4 + 20, h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    elif step == "success":
        cv2.putText(frame, f"{name} enregistre !", (w // 4 + 20, h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_KNOWN, 2)
    elif step == "error":
        cv2.putText(frame, "Aucun visage detecte.", (w // 4 + 20, h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_UNKNOWN, 2)

    return frame


def capture_face_encoding(cap):
    """Capture et encode un visage (moyenne de 5 captures)."""
    all_enc = []
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = frame[:, :, ::-1].copy()
        locs = face_recognition.face_locations(rgb, model="hog")
        if len(locs) > 0:
            largest = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
            enc = face_recognition.face_encodings(rgb, [largest])
            if len(enc) > 0:
                all_enc.append(enc[0])
        time.sleep(0.05)

    if len(all_enc) > 0:
        return np.mean(all_enc, axis=0)
    return None


def main():
    db = FaceDatabase()
    print(f"[INFO] DB: {db.count()} personne(s)")

    # ── Haar Cascade (détection rapide) ──
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[INFO] FPS webcam: {cap.get(cv2.CAP_PROP_FPS):.0f}")
    print("=" * 60)
    print("  DÉTECTION ET RECONNAISSANCE DE VISAGES")
    print("=" * 60)
    print("  [R] Enregistrer  [L] Lister  [D] Supprimer  [Q] Quitter")
    print("=" * 60)

    # Thread d'identification
    identifier = IdentificationThread(db)

    mode = "recognition"
    reg_name = ""
    reg_step = ""
    reg_timer = 0
    prev_tick = cv2.getTickCount()
    fps = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # ── Détection rapide avec Haar Cascade (chaque frame) ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # ── Envoyer 1 frame sur 5 au thread d'identification ──
        if frame_count % 5 == 0 and mode == "recognition":
            identifier.submit_frame(frame)

        # ── Récupérer les identités du thread ──
        identities = identifier.get_identities()

        known_count = 0
        unknown_count = 0

        for (x, y, w, h) in faces:
            name, color = match_face_to_identity((x, y, w, h), identities)

            if name is None:
                name = "..."
                color = COLOR_PENDING
            elif name == "Inconnu":
                unknown_count += 1
            else:
                known_count += 1

            frame = draw_face_box(frame, x, y, w, h, name, color)

        face_count = len(faces)

        # ── FPS ──
        current_tick = cv2.getTickCount()
        elapsed = (current_tick - prev_tick) / cv2.getTickFrequency()
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        prev_tick = current_tick

        # ── HUD ──
        frame = draw_hud(frame, face_count, known_count, unknown_count, fps, mode, db.count())

        if mode == "registration" and reg_step:
            frame = draw_registration_prompt(frame, reg_step, reg_name)
            if reg_step in ("success", "error") and time.time() - reg_timer > 2:
                mode = "recognition"
                reg_step = ""

        cv2.imshow("Face Detection", frame)

        # ── Touches ──
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        elif key in (ord("r"), ord("R")) and mode == "recognition":
            mode = "registration"
            reg_name = input("\n  Nom de la personne: ").strip()
            if reg_name:
                reg_step = "name_entered"
                print("  → Regardez la caméra, appuyez [ESPACE].")
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
                print("  ✗ Aucun visage détecté.")
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
                        print(f"  ✓ Supprimé.")
                except (ValueError, IndexError):
                    pass

    identifier.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Terminé.")


if __name__ == "__main__":
    main()