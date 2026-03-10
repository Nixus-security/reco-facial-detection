"""
Détection de visages en temps réel via webcam
- Détecte un ou plusieurs visages
- Applique un filtre visuel sur chaque visage détecté
- Affiche le nombre de personnes détectées
"""

import cv2
import numpy as np
import sys
import os


def create_overlay_mask(shape, color, alpha=0.3):
    """Crée un masque de couleur semi-transparent."""
    overlay = np.full(shape, color, dtype=np.uint8)
    return overlay, alpha


def apply_pixelate_filter(frame, x, y, w, h, pixel_size=12):
    """Applique un filtre pixelisé (mosaïque) sur une zone du visage."""
    face_roi = frame[y:y + h, x:x + w]
    small = cv2.resize(face_roi, (max(1, w // pixel_size), max(1, h // pixel_size)),
                       interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y + h, x:x + w] = pixelated
    return frame


def apply_color_overlay_filter(frame, x, y, w, h, color=(0, 200, 255), alpha=0.4):
    """Applique un filtre de couleur semi-transparent sur le visage."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def apply_edge_filter(frame, x, y, w, h):
    """Applique un filtre de détection de contours style néon sur le visage."""
    face_roi = frame[y:y + h, x:x + w]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_face, 80, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Colorer les contours en cyan néon
    edges_colored[edges > 0] = [255, 255, 0]
    # Mélanger avec le visage original
    blended = cv2.addWeighted(face_roi, 0.5, edges_colored, 0.8, 0)
    frame[y:y + h, x:x + w] = blended
    return frame


def apply_blur_filter(frame, x, y, w, h, blur_strength=35):
    """Applique un filtre de flou gaussien sur le visage."""
    # blur_strength doit être impair
    if blur_strength % 2 == 0:
        blur_strength += 1
    face_roi = frame[y:y + h, x:x + w]
    blurred = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 30)
    frame[y:y + h, x:x + w] = blurred
    return frame


def apply_cartoon_filter(frame, x, y, w, h):
    """Applique un filtre cartoon sur le visage."""
    face_roi = frame[y:y + h, x:x + w]
    # Réduction du bruit
    color = cv2.bilateralFilter(face_roi, 9, 250, 250)
    # Détection des contours
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Combiner couleur + contours
    cartoon = cv2.bitwise_and(color, edges_colored)
    frame[y:y + h, x:x + w] = cartoon
    return frame


# ─── Dictionnaire des filtres disponibles ───────────────────────────────────
FILTERS = {
    "pixelate": {
        "name": "Pixelisé (Mosaïque)",
        "func": apply_pixelate_filter,
        "color": (0, 255, 255),
    },
    "color": {
        "name": "Overlay Couleur",
        "func": apply_color_overlay_filter,
        "color": (0, 200, 255),
    },
    "edge": {
        "name": "Contours Néon",
        "func": apply_edge_filter,
        "color": (255, 255, 0),
    },
    "blur": {
        "name": "Flou Gaussien",
        "func": apply_blur_filter,
        "color": (255, 100, 100),
    },
    "cartoon": {
        "name": "Cartoon",
        "func": apply_cartoon_filter,
        "color": (100, 255, 100),
    },
}

FILTER_KEYS = list(FILTERS.keys())


def draw_hud(frame, face_count, current_filter_key, fps):
    """Dessine le HUD (interface) par-dessus le flux vidéo."""
    h, w = frame.shape[:2]
    filter_info = FILTERS[current_filter_key]

    # ── Bandeau supérieur ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Titre
    cv2.putText(frame, "FACE DETECTION", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    # Nombre de personnes détectées
    count_text = f"Personnes detectees: {face_count}"
    color = (0, 255, 0) if face_count > 0 else (0, 0, 255)
    cv2.putText(frame, count_text, (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Filtre actif
    filter_text = f"Filtre: {filter_info['name']}"
    cv2.putText(frame, filter_text, (w // 2 - 50, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, filter_info["color"], 1)

    # ── Bandeau inférieur (contrôles) ──
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    controls = "[F] Changer filtre  |  [+/-] Sensibilite  |  [Q/ESC] Quitter"
    cv2.putText(frame, controls, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return frame


def draw_face_box(frame, x, y, w, h, face_id, color):
    """Dessine un cadre stylisé autour du visage détecté."""
    corner_len = min(w, h) // 4
    thickness = 2

    # Coins stylisés (au lieu d'un rectangle plein)
    # Haut-gauche
    cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
    # Haut-droit
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
    # Bas-gauche
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
    # Bas-droit
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    # Label
    label = f"Visage #{face_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def main():
    # ── Charger le classifieur Haar Cascade ──
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print(f"[ERREUR] Fichier cascade introuvable: {cascade_path}")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(cascade_path)

    # ── Ouvrir la webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        print("  → Vérifiez que votre webcam est branchée et non utilisée par une autre app.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 55)
    print("  DÉTECTION DE VISAGES - TEMPS RÉEL")
    print("=" * 55)
    print("  [F]       → Changer de filtre")
    print("  [+/-]     → Ajuster la sensibilité de détection")
    print("  [Q / ESC] → Quitter")
    print("=" * 55)

    current_filter_idx = 0
    min_neighbors = 5  # Sensibilité (plus bas = plus sensible, plus de faux positifs)
    scale_factor = 1.15
    prev_tick = cv2.getTickCount()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Impossible de lire le flux webcam.")
            break

        # Miroir horizontal (plus naturel)
        frame = cv2.flip(frame, 1)

        # Convertir en niveaux de gris pour la détection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # ── Détection des visages ──
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        face_count = len(faces)
        current_filter_key = FILTER_KEYS[current_filter_idx]
        filter_info = FILTERS[current_filter_key]
        filter_func = filter_info["func"]
        box_color = filter_info["color"]

        # ── Appliquer filtre + cadre sur chaque visage ──
        for i, (x, y, w, h) in enumerate(faces):
            # Agrandir légèrement la zone de détection pour le filtre
            pad = int(0.1 * h)
            fx = max(0, x - pad)
            fy = max(0, y - pad)
            fw = min(frame.shape[1] - fx, w + 2 * pad)
            fh = min(frame.shape[0] - fy, h + 2 * pad)

            frame = filter_func(frame, fx, fy, fw, fh)
            frame = draw_face_box(frame, x, y, w, h, i + 1, box_color)

        # ── Calculer FPS ──
        current_tick = cv2.getTickCount()
        time_elapsed = (current_tick - prev_tick) / cv2.getTickFrequency()
        if time_elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / time_elapsed)
        prev_tick = current_tick

        # ── Dessiner le HUD ──
        frame = draw_hud(frame, face_count, current_filter_key, fps)

        # ── Afficher ──
        cv2.imshow("Face Detection", frame)

        # ── Gestion des touches ──
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # Q ou ESC
            break
        elif key in (ord("f"), ord("F")):
            current_filter_idx = (current_filter_idx + 1) % len(FILTER_KEYS)
            new_filter = FILTERS[FILTER_KEYS[current_filter_idx]]["name"]
            print(f"  ↳ Filtre changé: {new_filter}")
        elif key in (ord("+"), ord("=")):
            min_neighbors = min(min_neighbors + 1, 15)
            print(f"  ↳ Sensibilité: minNeighbors={min_neighbors} (moins sensible)")
        elif key in (ord("-"), ord("_")):
            min_neighbors = max(min_neighbors - 1, 1)
            print(f"  ↳ Sensibilité: minNeighbors={min_neighbors} (plus sensible)")

    # ── Nettoyage ──
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Programme terminé proprement.")


if __name__ == "__main__":
    main()