#!/usr/bin/env python3
"""
Import Swiss RAG Project to Kanboard
Importiert alle Arbeitspakete automatisch in Kanboard
"""

import requests
import json
import base64
from datetime import datetime, timedelta

# Kanboard API Configuration
KANBOARD_URL = "http://localhost:8000/jsonrpc.php"
KANBOARD_USER = "admin"
KANBOARD_PASSWORD = "admin"

# Sprint und Team Daten
SPRINTS = [
    "Sprint 1: Schweizer Marktbereitschaft (22.-28. Juli)",
    "Sprint 2: Enterprise-Features (29. Juli - 4. August)", 
    "Sprint 3: Vertriebstools (5.-11. August)",
    "Sprint 4: Launch-Vorbereitung (12.-18. August)",
    "Sprint 5: Markt-Launch (19.-25. August)",
    "Sprint 6: Optimierung & Wachstum (26. August - 1. September)"
]

TEAM_MEMBERS = ["Tech Lead", "Frontend Developer", "Product Manager"]

# Arbeitspakete (vereinfacht für Demo)
WORK_PACKAGES = {
    "Sprint 1": {
        "Tech Lead": [
            "TL-S1-001: Multi-Tenancy Grundlagen (2 Tage, Kritisch)",
            "TL-S1-002: Schweizer Compliance-Infrastruktur (2 Tage, Kritisch)", 
            "TL-S1-003: Produktions-Deployment-Pipeline (1 Tag, Hoch)"
        ],
        "Frontend Developer": [
            "FE-S1-001: Deutsche Lokalisierung (2 Tage, Kritisch)",
            "FE-S1-002: Schweizer Design System (1.5 Tage, Hoch)",
            "FE-S1-003: Datenschutz & Einverständnis-UI (1.5 Tage, Kritisch)"
        ],
        "Product Manager": [
            "PM-S1-001: Schweizer Marktforschung (2 Tage, Kritisch)",
            "PM-S1-002: Go-to-Market Strategie (2 Tage, Hoch)",
            "PM-S1-003: Rechtsdokumente (1 Tag, Kritisch)"
        ]
    },
    "Sprint 2": {
        "Tech Lead": [
            "TL-S2-001: Erweitertes Caching & Performance (2 Tage, Hoch)",
            "TL-S2-002: API-Versionierung & Enterprise-Integration (2 Tage, Hoch)",
            "TL-S2-003: Sicherheitshärtung & Audit-Vorbereitung (1 Tag, Kritisch)"
        ],
        "Frontend Developer": [
            "FE-S2-001: Kunden-Dashboard-Entwicklung (2.5 Tage, Hoch)",
            "FE-S2-002: Französische Lokalisierung & Barrierefreiheit (1.5 Tage, Mittel)"
        ],
        "Product Manager": [
            "PM-S2-001: Beta-Kundenprogramm-Launch (2 Tage, Kritisch)",
            "PM-S2-002: Schweizer Partnerschaftsentwicklung (2 Tage, Hoch)",
            "PM-S2-003: Preis- & Geschäftsmodell-Optimierung (1 Tag, Mittel)"
        ]
    },
    "Sprint 3": {
        "Tech Lead": [
            "TL-S3-001: Demo-Umgebung & Sales-Engineering-Tools (2 Tage, Hoch)",
            "TL-S3-002: Integrationsplattform & Partner-APIs (2 Tage, Hoch)",
            "TL-S3-003: Performance-Optimierung & Skalierungsvorbereitung (1 Tag, Mittel)"
        ],
        "Frontend Developer": [
            "FE-S3-001: Sales-Demo-Interface & Branchenanpassung (2 Tage, Hoch)",
            "FE-S3-002: Mobile Responsiveness & Progressive Web App (2 Tage, Mittel)",
            "FE-S3-003: Analytics & Reporting Dashboard (1 Tag, Mittel)"
        ],
        "Product Manager": [
            "PM-S3-001: Sales-Enablement & Channel-Entwicklung (2 Tage, Kritisch)",
            "PM-S3-002: Marktdurchdringung & Kundenakquisition (2 Tage, Kritisch)",
            "PM-S3-003: Produkt-Roadmap & Feature-Priorisierung (1 Tag, Mittel)"
        ]
    },
    "Sprint 4": {
        "Tech Lead": [
            "TL-S4-001: Produktionsbereitschaft & Launch-Infrastruktur (2 Tage, Kritisch)",
            "TL-S4-002: Kunden-Onboarding-Automatisierung (2 Tage, Hoch)",
            "TL-S4-003: API-Dokumentation & Entwicklererfahrung (1 Tag, Mittel)"
        ],
        "Frontend Developer": [
            "FE-S4-001: Launch-bereite UI-Politur & Optimierung (2 Tage, Hoch)",
            "FE-S4-002: Kunden-Support & Hilfesystem (1.5 Tage, Hoch)",
            "FE-S4-003: Marketing-Website & Landing Pages (1.5 Tage, Mittel)"
        ],
        "Product Manager": [
            "PM-S4-001: Launch-Kampagne & PR-Strategie (2 Tage, Kritisch)",
            "PM-S4-002: Kunden-Erfolg & Support-Operationen (2 Tage, Kritisch)",
            "PM-S4-003: Launch-Metriken & Erfolgsmessung (1 Tag, Hoch)"
        ]
    },
    "Sprint 5": {
        "Tech Lead": [
            "TL-S5-001: Launch-Tag-Operationen & Monitoring (3 Tage, Kritisch)",
            "TL-S5-002: Kunden-Onboarding-Support & technische Unterstützung (2 Tage, Hoch)"
        ],
        "Frontend Developer": [
            "FE-S5-001: Launch-Woche UI-Support & Optimierung (3 Tage, Hoch)",
            "FE-S5-002: Marketing-Website-Performance & Konversionsoptimierung (2 Tage, Mittel)"
        ],
        "Product Manager": [
            "PM-S5-001: Launch-Durchführung & Kampagnenmanagement (3 Tage, Kritisch)",
            "PM-S5-002: Kundenakquisition & Sales-Pipeline-Management (2 Tage, Kritisch)"
        ]
    },
    "Sprint 6": {
        "Tech Lead": [
            "TL-S6-001: Post-Launch-Optimierung & Performance-Tuning (2 Tage, Hoch)",
            "TL-S6-002: Kunden-Feedback-Integration & Feature-Entwicklung (2 Tage, Mittel)",
            "TL-S6-003: Sicherheit & Compliance-Verbesserung (1 Tag, Hoch)"
        ],
        "Frontend Developer": [
            "FE-S6-001: User-Experience-Optimierung & A/B-Testing (2 Tage, Hoch)",
            "FE-S6-002: Erweiterte Features & kundenangeforderte Verbesserungen (2 Tage, Mittel)",
            "FE-S6-003: Mobile-App-Entwicklungsplanung (1 Tag, Niedrig)"
        ],
        "Product Manager": [
            "PM-S6-001: Wachstumsstrategie & Marktexpansionsplanung (2 Tage, Kritisch)",
            "PM-S6-002: Partnerschaft & Channel-Entwicklung (2 Tage, Hoch)",
            "PM-S6-003: Produkt-Roadmap & Zukunftsplanung (1 Tag, Mittel)"
        ]
    }
}

class KanboardAPI:
    def __init__(self, url, username, password):
        self.url = url
        self.auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {self.auth}'
        }
        
    def call_api(self, method, params=None):
        """Kanboard API Aufruf"""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": 1
        }
        if params:
            payload["params"] = params
            
        response = requests.post(self.url, json=payload, headers=self.headers)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('result')
        else:
            print(f"❌ API Fehler: {response.text}")
            return None
    
    def create_project(self, name, description=""):
        """Projekt erstellen"""
        return self.call_api("createProject", {
            "name": name,
            "description": description
        })
    
    def create_swimlane(self, project_id, name):
        """Swimlane erstellen (für Team-Mitglieder)"""
        return self.call_api("addSwimlane", {
            "project_id": project_id,
            "name": name
        })
    
    def create_column(self, project_id, title, position):
        """Spalte erstellen"""
        return self.call_api("addColumn", {
            "project_id": project_id,
            "title": title,
            "task_limit": 0,
            "position": position
        })
    
    def create_task(self, project_id, title, column_id, swimlane_id=0, color="blue"):
        """Task erstellen"""
        return self.call_api("createTask", {
            "title": title,
            "project_id": project_id,
            "column_id": column_id,
            "swimlane_id": swimlane_id,
            "color_id": color
        })

def setup_kanboard_project():
    """Swiss RAG Project in Kanboard einrichten"""
    print("🚀 Swiss RAG Project Setup in Kanboard")
    print("=" * 50)
    
    api = KanboardAPI(KANBOARD_URL, KANBOARD_USER, KANBOARD_PASSWORD)
    
    # 1. Projekt erstellen
    print("📋 Erstelle Hauptprojekt...")
    project_id = api.create_project(
        "Swiss RAG System Launch",
        "6-Wochen Sprint Plan für Schweizer Markteinführung"
    )
    
    if not project_id:
        print("❌ Projekt konnte nicht erstellt werden!")
        return
    
    print(f"✅ Projekt erstellt (ID: {project_id})")
    
    # 2. Spalten für Sprints erstellen (Standard-Spalten ersetzen)
    print("\n📅 Erstelle Sprint-Spalten...")
    
    # Erst Standard-Spalten holen
    columns = api.call_api("getColumns", {"project_id": project_id})
    
    # Neue Spalten für Sprints hinzufügen
    for i, sprint in enumerate(SPRINTS[:4]):  # Nur erste 4 Sprints als Spalten
        api.create_column(project_id, sprint, i + 5)
        print(f"  ✅ {sprint}")
    
    # 3. Swimlanes für Team-Mitglieder erstellen
    print("\n👥 Erstelle Team-Swimlanes...")
    swimlane_ids = {}
    
    for member in TEAM_MEMBERS:
        swimlane_id = api.create_swimlane(project_id, member)
        swimlane_ids[member] = swimlane_id
        print(f"  ✅ {member} (ID: {swimlane_id})")
    
    # 4. Tasks für Sprint 1 erstellen (Beispiel)
    print("\n📝 Erstelle Arbeitspakete für Sprint 1...")
    
    # Sprint 1 Spalte finden
    columns = api.call_api("getColumns", {"project_id": project_id})
    sprint1_column = None
    for col in columns:
        if "Sprint 1" in col["title"]:
            sprint1_column = col["id"]
            break
    
    if sprint1_column:
        sprint1_tasks = WORK_PACKAGES["Sprint 1"]
        
        for member, tasks in sprint1_tasks.items():
            member_swimlane = swimlane_ids.get(member, 0)
            
            for task in tasks:
                # Farbe basierend auf Priorität
                color = "red" if "Kritisch" in task else "orange" if "Hoch" in task else "blue"
                
                task_id = api.create_task(
                    project_id=project_id,
                    title=task,
                    column_id=sprint1_column,
                    swimlane_id=member_swimlane,
                    color=color
                )
                
                if task_id:
                    print(f"  ✅ {task[:50]}...")
    
    print("\n🎉 Setup abgeschlossen!")
    print(f"🔗 Kanboard öffnen: http://localhost:8000")
    print(f"📋 Projekt: Swiss RAG System Launch")
    print("\n💡 Nächste Schritte:")
    print("   1. Weitere Sprints manuell als Spalten hinzufügen")
    print("   2. Restliche Arbeitspakete als Tasks erstellen")
    print("   3. Due Dates für Tasks setzen")
    print("   4. Team-Mitglieder einladen (falls Multi-User Setup)")

if __name__ == "__main__":
    setup_kanboard_project()