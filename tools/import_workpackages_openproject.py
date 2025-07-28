#!/usr/bin/env python3
"""
Import Work Packages to OpenProject
Importiert die deutschen Arbeitspakete in OpenProject via API
"""

import json
import requests
from datetime import datetime, timedelta
import os

# OpenProject API Configuration
OPENPROJECT_URL = os.getenv('OPENPROJECT_URL', 'http://localhost:8080')
API_KEY = os.getenv('OPENPROJECT_API_KEY', 'your-api-key-here')

# Sprint-Daten
SPRINTS = [
    {
        "name": "Sprint 1: Schweizer Marktbereitschaft",
        "start_date": "2025-07-22",
        "end_date": "2025-07-28"
    },
    {
        "name": "Sprint 2: Enterprise-Features",
        "start_date": "2025-07-29", 
        "end_date": "2025-08-04"
    },
    {
        "name": "Sprint 3: Vertriebstools",
        "start_date": "2025-08-05",
        "end_date": "2025-08-11"
    },
    {
        "name": "Sprint 4: Launch-Vorbereitung",
        "start_date": "2025-08-12",
        "end_date": "2025-08-18"
    },
    {
        "name": "Sprint 5: Markt-Launch", 
        "start_date": "2025-08-19",
        "end_date": "2025-08-25"
    },
    {
        "name": "Sprint 6: Optimierung & Wachstum",
        "start_date": "2025-08-26",
        "end_date": "2025-09-01"
    }
]

# Beispiel-Arbeitspakete (vereinfacht)
WORK_PACKAGES = {
    "Sprint 1": {
        "Tech Lead": [
            {
                "id": "TL-S1-001",
                "title": "Multi-Tenancy Grundlagen",
                "duration": 2,
                "priority": "Kritisch",
                "description": "Mandantenfähige Datenbankarchitektur entwickeln"
            },
            {
                "id": "TL-S1-002", 
                "title": "Schweizer Compliance-Infrastruktur",
                "duration": 2,
                "priority": "Kritisch",
                "description": "DSGVO/FADP Datenexport-Funktionalität implementieren"
            },
            {
                "id": "TL-S1-003",
                "title": "Produktions-Deployment-Pipeline",
                "duration": 1,
                "priority": "Hoch",
                "description": "Schweizer Rechenzentrum-Hosting-Umgebung konfigurieren"
            }
        ],
        "Frontend Developer": [
            {
                "id": "FE-S1-001",
                "title": "Deutsche Lokalisierung",
                "duration": 2,
                "priority": "Kritisch",
                "description": "i18n Framework implementieren und deutsche Übersetzung"
            },
            {
                "id": "FE-S1-002",
                "title": "Schweizer Design System",
                "duration": 1.5,
                "priority": "Hoch",
                "description": "Schweiz-inspirierte UI-Komponenten erstellen"
            },
            {
                "id": "FE-S1-003",
                "title": "Datenschutz & Einverständnis-UI",
                "duration": 1.5,
                "priority": "Kritisch",
                "description": "DSGVO/FADP-konforme Datenschutz-Banner erstellen"
            }
        ],
        "Product Manager": [
            {
                "id": "PM-S1-001",
                "title": "Schweizer Marktforschung",
                "duration": 2,
                "priority": "Kritisch",
                "description": "5 Kundeninterviews mit Schweizer Unternehmen"
            },
            {
                "id": "PM-S1-002",
                "title": "Go-to-Market Strategie",
                "duration": 2,
                "priority": "Hoch",
                "description": "Schweizer Vertriebs- und Marketingstrategie entwickeln"
            },
            {
                "id": "PM-S1-003",
                "title": "Rechtsdokumente",
                "duration": 1,
                "priority": "Kritisch",
                "description": "Deutsche AGB und Datenschutzerklärung erstellen"
            }
        ]
    }
    # Weitere Sprints würden hier folgen...
}

class OpenProjectImporter:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            'apikey': api_key,
            'Content-Type': 'application/json'
        }
        
    def create_project(self, name, description):
        """Erstellt ein neues Projekt in OpenProject"""
        url = f"{self.base_url}/api/v3/projects"
        data = {
            "name": name,
            "identifier": name.lower().replace(' ', '-'),
            "description": {"raw": description, "format": "markdown"},
            "public": False
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code == 201:
            print(f"✅ Projekt '{name}' erfolgreich erstellt")
            return response.json()
        else:
            print(f"❌ Fehler beim Erstellen des Projekts: {response.text}")
            return None
            
    def create_version(self, project_id, sprint_data):
        """Erstellt eine Version (Sprint) im Projekt"""
        url = f"{self.base_url}/api/v3/projects/{project_id}/versions"
        data = {
            "name": sprint_data["name"],
            "startDate": sprint_data["start_date"],
            "endDate": sprint_data["end_date"],
            "status": "open"
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code == 201:
            print(f"✅ Sprint '{sprint_data['name']}' erstellt")
            return response.json()
        else:
            print(f"❌ Fehler beim Sprint erstellen: {response.text}")
            return None
            
    def create_work_package(self, project_id, version_id, package_data, assignee):
        """Erstellt ein Arbeitspaket"""
        url = f"{self.base_url}/api/v3/work_packages"
        
        priority_map = {
            "Kritisch": 1,
            "Hoch": 2,
            "Mittel": 3,
            "Niedrig": 4
        }
        
        data = {
            "subject": f"[{package_data['id']}] {package_data['title']}",
            "description": {
                "raw": package_data['description'],
                "format": "markdown"
            },
            "project": {"href": f"/api/v3/projects/{project_id}"},
            "type": {"href": "/api/v3/types/1"},  # Task
            "status": {"href": "/api/v3/statuses/1"},  # New
            "priority": {"href": f"/api/v3/priorities/{priority_map.get(package_data['priority'], 3)}"},
            "estimatedTime": f"PT{int(package_data['duration'] * 8)}H",  # Tage zu Stunden
            "assignee": {"href": f"/api/v3/users/{assignee}"},
            "version": {"href": f"/api/v3/versions/{version_id}"}
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code == 201:
            print(f"  ✅ Arbeitspaket '{package_data['id']}' erstellt")
            return response.json()
        else:
            print(f"  ❌ Fehler beim Arbeitspaket: {response.text}")
            return None

def main():
    """Hauptfunktion zum Importieren der Arbeitspakete"""
    print("🚀 OpenProject Work Package Import")
    print("==================================")
    
    # API-Verbindung prüfen
    importer = OpenProjectImporter(OPENPROJECT_URL, API_KEY)
    
    # Projekt erstellen
    project_name = "Swiss RAG System Launch"
    project_desc = """
    # Swiss RAG System Market Launch
    
    6-Wochen-Sprint-Plan für die Einführung des RAG Systems im Schweizer Markt.
    
    **Timeline**: 22. Juli - 1. September 2025
    **Team**: Tech Lead, Frontend Developer, Product Manager
    **Ziel**: Erfolgreiche Markteinführung in der Schweiz
    """
    
    project = importer.create_project(project_name, project_desc)
    if not project:
        return
        
    project_id = project['id']
    
    # Benutzer-IDs (müssen angepasst werden)
    users = {
        "Tech Lead": 2,
        "Frontend Developer": 3,
        "Product Manager": 4
    }
    
    # Sprints und Arbeitspakete erstellen
    for i, sprint in enumerate(SPRINTS):
        print(f"\n📅 Erstelle {sprint['name']}...")
        version = importer.create_version(project_id, sprint)
        
        if version and f"Sprint {i+1}" in WORK_PACKAGES:
            version_id = version['id']
            sprint_packages = WORK_PACKAGES[f"Sprint {i+1}"]
            
            for role, packages in sprint_packages.items():
                print(f"\n  👤 {role} Arbeitspakete:")
                for package in packages:
                    importer.create_work_package(
                        project_id, 
                        version_id, 
                        package, 
                        users.get(role, 1)
                    )
    
    print("\n✅ Import abgeschlossen!")
    print(f"🔗 Projekt ansehen: {OPENPROJECT_URL}/projects/{project_id}")

if __name__ == "__main__":
    main()