"""
Module de base de données pour la reconnaissance faciale.
Stocke les encodages de visages dans un fichier JSON local.
"""

import json
import os
import numpy as np

DB_FILE = "face_database.json"


class FaceDatabase:
    """Gère le stockage et la récupération des encodages de visages."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_FILE
        self.data = {}  # { "nom": [encoding_list] }
        self._load()

    def _load(self):
        """Charge la base de données depuis le fichier JSON."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Convertir les listes en numpy arrays
                self.data = {
                    name: np.array(encoding)
                    for name, encoding in raw.items()
                }
            except (json.JSONDecodeError, Exception) as e:
                print(f"[WARN] Erreur lecture DB, création d'une nouvelle: {e}")
                self.data = {}
        else:
            self.data = {}

    def _save(self):
        """Sauvegarde la base de données dans le fichier JSON."""
        raw = {
            name: encoding.tolist()
            for name, encoding in self.data.items()
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

    def add_person(self, name, encoding):
        """Ajoute ou met à jour une personne dans la base."""
        self.data[name] = np.array(encoding)
        self._save()

    def remove_person(self, name):
        """Supprime une personne de la base."""
        if name in self.data:
            del self.data[name]
            self._save()
            return True
        return False

    def get_all_encodings(self):
        """Retourne tous les encodages sous forme de liste numpy."""
        if len(self.data) == 0:
            return []
        return list(self.data.values())

    def get_all_names(self):
        """Retourne tous les noms enregistrés."""
        return list(self.data.keys())

    def list_persons(self):
        """Retourne la liste des noms enregistrés."""
        return list(self.data.keys())

    def count(self):
        """Retourne le nombre de personnes enregistrées."""
        return len(self.data)

    def exists(self, name):
        """Vérifie si une personne est enregistrée."""
        return name in self.data