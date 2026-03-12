"""
Module de base de données pour la reconnaissance faciale.
Stocke les encodages de visages et le mot de passe de déverrouillage.
"""

import json
import os
import hashlib
import numpy as np

DB_FILE = "face_database.json"


class FaceDatabase:
    """Gère le stockage des encodages de visages et du mot de passe."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_FILE
        self.data = {}
        self.password_hash = None
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Extraire le mot de passe s'il existe
                self.password_hash = raw.pop("__password_hash__", None)
                # Convertir les listes en numpy arrays
                self.data = {
                    name: np.array(encoding)
                    for name, encoding in raw.items()
                }
            except (json.JSONDecodeError, Exception) as e:
                print(f"[WARN] Erreur lecture DB: {e}")
                self.data = {}
        else:
            self.data = {}

    def _save(self):
        raw = {
            name: encoding.tolist()
            for name, encoding in self.data.items()
        }
        if self.password_hash:
            raw["__password_hash__"] = self.password_hash
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

    def set_password(self, password):
        """Définit le mot de passe de déverrouillage."""
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
        self._save()

    def check_password(self, password):
        """Vérifie le mot de passe."""
        if self.password_hash is None:
            return False
        return hashlib.sha256(password.encode()).hexdigest() == self.password_hash

    def has_password(self):
        """Vérifie si un mot de passe est configuré."""
        return self.password_hash is not None

    def add_person(self, name, encoding):
        self.data[name] = np.array(encoding)
        self._save()

    def remove_person(self, name):
        if name in self.data:
            del self.data[name]
            self._save()
            return True
        return False

    def get_all_encodings(self):
        if len(self.data) == 0:
            return []
        return list(self.data.values())

    def get_all_names(self):
        return list(self.data.keys())

    def list_persons(self):
        return list(self.data.keys())

    def count(self):
        return len(self.data)

    def exists(self, name):
        return name in self.data