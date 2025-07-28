#!/usr/bin/env python3
"""
Creates Excel file with German work packages for Swiss RAG System Launch
Erstellt Excel-Datei mit deutschen Arbeitspaketen für Schweizer RAG-Launch
"""

import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import configuration
try:
    from config.config import config
    CONFIG_AVAILABLE = True
except ImportError:
    config = None
    CONFIG_AVAILABLE = False

# Sprint-Daten
SPRINTS = [
    {"name": "Sprint 1: Schweizer Marktbereitschaft", "start": "2025-07-22", "end": "2025-07-28"},
    {"name": "Sprint 2: Enterprise-Features", "start": "2025-07-29", "end": "2025-08-04"},
    {"name": "Sprint 3: Vertriebstools", "start": "2025-08-05", "end": "2025-08-11"},
    {"name": "Sprint 4: Launch-Vorbereitung", "start": "2025-08-12", "end": "2025-08-18"},
    {"name": "Sprint 5: Markt-Launch", "start": "2025-08-19", "end": "2025-08-25"},
    {"name": "Sprint 6: Optimierung & Wachstum", "start": "2025-08-26", "end": "2025-09-01"}
]

# Arbeitspakete-Daten
WORK_PACKAGES = [
    # Sprint 1
    {"Sprint": "Sprint 1", "Paket_ID": "TL-S1-001", "Rolle": "Tech Lead", "Titel": "Multi-Tenancy Grundlagen", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "Mandantenfähige Datenbankarchitektur entwickeln", "Liefergegenstände": "Tenant-Middleware, DB-Migrationen, Unit-Tests", "Status": "Geplant"},
    {"Sprint": "Sprint 1", "Paket_ID": "TL-S1-002", "Rolle": "Tech Lead", "Titel": "Schweizer Compliance-Infrastruktur", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "DSGVO/FADP Datenexport-Funktionalität implementieren", "Liefergegenstände": "Compliance-Service, Audit-Logging, Datenretention", "Status": "Geplant"},
    {"Sprint": "Sprint 1", "Paket_ID": "TL-S1-003", "Rolle": "Tech Lead", "Titel": "Produktions-Deployment-Pipeline", "Dauer_Tage": 1, "Priorität": "Hoch", "Beschreibung": "Schweizer Rechenzentrum-Hosting-Umgebung konfigurieren", "Liefergegenstände": "CI/CD Pipeline, Monitoring, Disaster Recovery", "Status": "Geplant"},
    
    {"Sprint": "Sprint 1", "Paket_ID": "FE-S1-001", "Rolle": "Frontend Developer", "Titel": "Deutsche Lokalisierung", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "i18n Framework implementieren und deutsche Übersetzung", "Liefergegenstände": "Deutsche Übersetzungen, Sprachumschaltung, CHF-Formatierung", "Status": "Geplant"},
    {"Sprint": "Sprint 1", "Paket_ID": "FE-S1-002", "Rolle": "Frontend Developer", "Titel": "Schweizer Design System", "Dauer_Tage": 1.5, "Priorität": "Hoch", "Beschreibung": "Schweiz-inspirierte UI-Komponenten erstellen", "Liefergegenstände": "Design System, Mobile Layouts, Barrierefreiheit", "Status": "Geplant"},
    {"Sprint": "Sprint 1", "Paket_ID": "FE-S1-003", "Rolle": "Frontend Developer", "Titel": "Datenschutz & Einverständnis-UI", "Dauer_Tage": 1.5, "Priorität": "Kritisch", "Beschreibung": "DSGVO/FADP-konforme Datenschutz-Banner erstellen", "Liefergegenstände": "Privacy Banner, Consent Management, Cookie Präferenzen", "Status": "Geplant"},
    
    {"Sprint": "Sprint 1", "Paket_ID": "PM-S1-001", "Rolle": "Product Manager", "Titel": "Schweizer Marktforschung", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "5 Kundeninterviews mit Schweizer Unternehmen", "Liefergegenstände": "Interview-Berichte, Konkurrenzanalyse, Kundenpersonas", "Status": "Geplant"},
    {"Sprint": "Sprint 1", "Paket_ID": "PM-S1-002", "Rolle": "Product Manager", "Titel": "Go-to-Market Strategie", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Schweizer Vertriebs- und Marketingstrategie entwickeln", "Liefergegenstände": "GTM-Strategie, Partner-Liste, Onboarding-Playbook", "Status": "Geplant"},
    {"Sprint": "Sprint 1", "Paket_ID": "PM-S1-003", "Rolle": "Product Manager", "Titel": "Rechtsdokumente", "Dauer_Tage": 1, "Priorität": "Kritisch", "Beschreibung": "Deutsche AGB und Datenschutzerklärung erstellen", "Liefergegenstände": "AGB, Datenschutz, DPA, Compliance-Materialien", "Status": "Geplant"},
    
    # Sprint 2
    {"Sprint": "Sprint 2", "Paket_ID": "TL-S2-001", "Rolle": "Tech Lead", "Titel": "Erweitertes Caching & Performance", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Redis-Cluster für Schweizer Rechenzentren implementieren", "Liefergegenstände": "Redis Caching, DB-Optimierung, Performance-Monitoring", "Status": "Geplant"},
    {"Sprint": "Sprint 2", "Paket_ID": "TL-S2-002", "Rolle": "Tech Lead", "Titel": "API-Versionierung & Enterprise-Integration", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "API v2 mit Rückwärtskompatibilität entwickeln", "Liefergegenstände": "API v2, Webhook-System, SSO-Integration", "Status": "Geplant"},
    {"Sprint": "Sprint 2", "Paket_ID": "TL-S2-003", "Rolle": "Tech Lead", "Titel": "Sicherheitshärtung & Audit-Vorbereitung", "Dauer_Tage": 1, "Priorität": "Kritisch", "Beschreibung": "Zusätzliche Sicherheitsschichten implementieren", "Liefergegenstände": "Security Audit, OWASP Compliance, Incident Response", "Status": "Geplant"},
    
    {"Sprint": "Sprint 2", "Paket_ID": "FE-S2-001", "Rolle": "Frontend Developer", "Titel": "Kunden-Dashboard-Entwicklung", "Dauer_Tage": 2.5, "Priorität": "Hoch", "Beschreibung": "Nutzungsanalyse-Dashboard entwickeln", "Liefergegenstände": "Dashboard, Dokumenten-Management, Team-UI, Billing", "Status": "Geplant"},
    {"Sprint": "Sprint 2", "Paket_ID": "FE-S2-002", "Rolle": "Frontend Developer", "Titel": "Französische Lokalisierung & Barrierefreiheit", "Dauer_Tage": 1.5, "Priorität": "Mittel", "Beschreibung": "Französische Übersetzungen erstellen", "Liefergegenstände": "Französische Übersetzungen, WCAG 2.1 AA, Hilfe-System", "Status": "Geplant"},
    
    {"Sprint": "Sprint 2", "Paket_ID": "PM-S2-001", "Rolle": "Product Manager", "Titel": "Beta-Kundenprogramm-Launch", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "5 Beta-Kunden aus Zielsegmenten rekrutieren", "Liefergegenstände": "5 Beta-Kunden, Feedback-System, Success-Tracking", "Status": "Geplant"},
    {"Sprint": "Sprint 2", "Paket_ID": "PM-S2-002", "Rolle": "Product Manager", "Titel": "Schweizer Partnerschaftsentwicklung", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Outreach zu Top-5 Schweizer Systemintegratoren", "Liefergegenstände": "2+ Partner-Verträge, Training-Programm, Joint-Marketing", "Status": "Geplant"},
    {"Sprint": "Sprint 2", "Paket_ID": "PM-S2-003", "Rolle": "Product Manager", "Titel": "Preis- & Geschäftsmodell-Optimierung", "Dauer_Tage": 1, "Priorität": "Mittel", "Beschreibung": "Beta-Kunden Zahlungsbereitschaft analysieren", "Liefergegenstände": "Preisstrategie, Enterprise-Framework, Upselling-Plan", "Status": "Geplant"},
    
    # Sprint 3
    {"Sprint": "Sprint 3", "Paket_ID": "TL-S3-001", "Rolle": "Tech Lead", "Titel": "Demo-Umgebung & Sales-Engineering-Tools", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Branchenspezifische Demo-Umgebungen erstellen", "Liefergegenstände": "3 Demo-Umgebungen, Sales-Tools, POC-Framework", "Status": "Geplant"},
    {"Sprint": "Sprint 3", "Paket_ID": "TL-S3-002", "Rolle": "Tech Lead", "Titel": "Integrationsplattform & Partner-APIs", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Integration für Schweizer Enterprise-Software", "Liefergegenstände": "Integrations-Plattform, Partner-APIs, Banking-Integration", "Status": "Geplant"},
    {"Sprint": "Sprint 3", "Paket_ID": "TL-S3-003", "Rolle": "Tech Lead", "Titel": "Performance-Optimierung & Skalierungsvorbereitung", "Dauer_Tage": 1, "Priorität": "Mittel", "Beschreibung": "System für 1000+ Benutzer optimieren", "Liefergegenstände": "Performance-Optimierung, Auto-Scaling, Load-Testing", "Status": "Geplant"},
    
    {"Sprint": "Sprint 3", "Paket_ID": "FE-S3-001", "Rolle": "Frontend Developer", "Titel": "Sales-Demo-Interface & Branchenanpassung", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Sales-Demo mit branchenspezifischen Themes", "Liefergegenstände": "Demo-Interface, Produkttour, Branchen-Templates", "Status": "Geplant"},
    {"Sprint": "Sprint 3", "Paket_ID": "FE-S3-002", "Rolle": "Frontend Developer", "Titel": "Mobile Responsiveness & Progressive Web App", "Dauer_Tage": 2, "Priorität": "Mittel", "Beschreibung": "Mobile und PWA Optimierung", "Liefergegenstände": "Mobile Design, PWA, Offline-Funktionen", "Status": "Geplant"},
    {"Sprint": "Sprint 3", "Paket_ID": "FE-S3-003", "Rolle": "Frontend Developer", "Titel": "Analytics & Reporting Dashboard", "Dauer_Tage": 1, "Priorität": "Mittel", "Beschreibung": "Analytics- und Reporting-Dashboard bauen", "Liefergegenstände": "Analytics-Dashboard, Datenvisualisierung, Export", "Status": "Geplant"},
    
    {"Sprint": "Sprint 3", "Paket_ID": "PM-S3-001", "Rolle": "Product Manager", "Titel": "Sales-Enablement & Channel-Entwicklung", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "Sales-Playbook und Training erstellen", "Liefergegenstände": "Sales-Playbook, Battlecards, Partner-Training, Case Studies", "Status": "Geplant"},
    {"Sprint": "Sprint 3", "Paket_ID": "PM-S3-002", "Rolle": "Product Manager", "Titel": "Marktdurchdringung & Kundenakquisition", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "Outreach zu 50+ Schweizer Enterprise-Prospects", "Liefergegenstände": "50+ Prospects, Konferenzen, Content-Marketing, Leads", "Status": "Geplant"},
    {"Sprint": "Sprint 3", "Paket_ID": "PM-S3-003", "Rolle": "Product Manager", "Titel": "Produkt-Roadmap & Feature-Priorisierung", "Dauer_Tage": 1, "Priorität": "Mittel", "Beschreibung": "Roadmap basierend auf Kundenfeedback priorisieren", "Liefergegenstände": "Roadmap, Feature-Specs, Ressourcenplan", "Status": "Geplant"},
    
    # Sprint 4
    {"Sprint": "Sprint 4", "Paket_ID": "TL-S4-001", "Rolle": "Tech Lead", "Titel": "Produktionsbereitschaft & Launch-Infrastruktur", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "Finale Produktionsumgebung Setup", "Liefergegenstände": "Produktions-Setup, Monitoring, Disaster Recovery", "Status": "Geplant"},
    {"Sprint": "Sprint 4", "Paket_ID": "TL-S4-002", "Rolle": "Tech Lead", "Titel": "Kunden-Onboarding-Automatisierung", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Automatisiertes Kunden-Onboarding System", "Liefergegenstände": "Onboarding-Platform, Self-Service, Health-Scoring", "Status": "Geplant"},
    {"Sprint": "Sprint 4", "Paket_ID": "TL-S4-003", "Rolle": "Tech Lead", "Titel": "API-Dokumentation & Entwicklererfahrung", "Dauer_Tage": 1, "Priorität": "Mittel", "Beschreibung": "Umfassende API-Dokumentation erstellen", "Liefergegenstände": "API-Docs, Explorer, SDKs, Developer Portal", "Status": "Geplant"},
    
    {"Sprint": "Sprint 4", "Paket_ID": "FE-S4-001", "Rolle": "Frontend Developer", "Titel": "Launch-bereite UI-Politur & Optimierung", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "UI für Launch-Qualität polieren", "Liefergegenstände": "Polierte UI, Performance-Optimierung, Cross-Browser Tests", "Status": "Geplant"},
    {"Sprint": "Sprint 4", "Paket_ID": "FE-S4-002", "Rolle": "Frontend Developer", "Titel": "Kunden-Support & Hilfesystem", "Dauer_Tage": 1.5, "Priorität": "Hoch", "Beschreibung": "In-App-Hilfe und Support-Integration", "Liefergegenstände": "In-App Hilfe, Tutorials, Support-Chat, Feedback", "Status": "Geplant"},
    {"Sprint": "Sprint 4", "Paket_ID": "FE-S4-003", "Rolle": "Frontend Developer", "Titel": "Marketing-Website & Landing Pages", "Dauer_Tage": 1.5, "Priorität": "Mittel", "Beschreibung": "Schweizer Marketing-Website erstellen", "Liefergegenstände": "Marketing-Website, Landing Pages, Lead-Capture, SEO", "Status": "Geplant"},
    
    {"Sprint": "Sprint 4", "Paket_ID": "PM-S4-001", "Rolle": "Product Manager", "Titel": "Launch-Kampagne & PR-Strategie", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "Schweizer Markt-Launch-Kampagne durchführen", "Liefergegenstände": "Launch-Kampagne, Medien-Coverage, Event, Content", "Status": "Geplant"},
    {"Sprint": "Sprint 4", "Paket_ID": "PM-S4-002", "Rolle": "Product Manager", "Titel": "Kunden-Erfolg & Support-Operationen", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "Customer Success Operations etablieren", "Liefergegenstände": "CS-Framework, Support-Training, Community", "Status": "Geplant"},
    {"Sprint": "Sprint 4", "Paket_ID": "PM-S4-003", "Rolle": "Product Manager", "Titel": "Launch-Metriken & Erfolgsmessung", "Dauer_Tage": 1, "Priorität": "Hoch", "Beschreibung": "Launch-KPIs und Monitoring etablieren", "Liefergegenstände": "KPI-Framework, Dashboard, Optimierungsplan", "Status": "Geplant"},
    
    # Sprint 5
    {"Sprint": "Sprint 5", "Paket_ID": "TL-S5-001", "Rolle": "Tech Lead", "Titel": "Launch-Tag-Operationen & Monitoring", "Dauer_Tage": 3, "Priorität": "Kritisch", "Beschreibung": "Launch-Tag technische Operationen", "Liefergegenstände": "Launch-Operations, Real-time Monitoring, Issue Resolution", "Status": "Geplant"},
    {"Sprint": "Sprint 5", "Paket_ID": "TL-S5-002", "Rolle": "Tech Lead", "Titel": "Kunden-Onboarding-Support & technische Unterstützung", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Technischer Support für neue Kunden", "Liefergegenstände": "Tech Support, Integration-Hilfe, Performance-Tuning", "Status": "Geplant"},
    
    {"Sprint": "Sprint 5", "Paket_ID": "FE-S5-001", "Rolle": "Frontend Developer", "Titel": "Launch-Woche UI-Support & Optimierung", "Dauer_Tage": 3, "Priorität": "Hoch", "Beschreibung": "UI-Performance während Launch überwachen", "Liefergegenstände": "UI-Monitoring, Bug-Fixes, UX-Optimierung", "Status": "Geplant"},
    {"Sprint": "Sprint 5", "Paket_ID": "FE-S5-002", "Rolle": "Frontend Developer", "Titel": "Marketing-Website-Performance & Konversionsoptimierung", "Dauer_Tage": 2, "Priorität": "Mittel", "Beschreibung": "Website-Performance während Launch", "Liefergegenstände": "Website-Optimierung, A/B-Tests, Conversion-Tuning", "Status": "Geplant"},
    
    {"Sprint": "Sprint 5", "Paket_ID": "PM-S5-001", "Rolle": "Product Manager", "Titel": "Launch-Durchführung & Kampagnenmanagement", "Dauer_Tage": 3, "Priorität": "Kritisch", "Beschreibung": "Launch-Kampagne über alle Kanäle", "Liefergegenstände": "Launch-Execution, Medien-Koordination, Event-Management", "Status": "Geplant"},
    {"Sprint": "Sprint 5", "Paket_ID": "PM-S5-002", "Rolle": "Product Manager", "Titel": "Kundenakquisition & Sales-Pipeline-Management", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "Sales-Pipeline während Launch verwalten", "Liefergegenstände": "Pipeline-Management, Lead-Support, CAC-Optimierung", "Status": "Geplant"},
    
    # Sprint 6
    {"Sprint": "Sprint 6", "Paket_ID": "TL-S6-001", "Rolle": "Tech Lead", "Titel": "Post-Launch-Optimierung & Performance-Tuning", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Launch-Performance analysieren und optimieren", "Liefergegenstände": "Performance-Analyse, System-Optimierung, Roadmap", "Status": "Geplant"},
    {"Sprint": "Sprint 6", "Paket_ID": "TL-S6-002", "Rolle": "Tech Lead", "Titel": "Kunden-Feedback-Integration & Feature-Entwicklung", "Dauer_Tage": 2, "Priorität": "Mittel", "Beschreibung": "Kundenfeedback analysieren und Features entwickeln", "Liefergegenstände": "Feedback-Analyse, Quick-Wins, Feature-Plan", "Status": "Geplant"},
    {"Sprint": "Sprint 6", "Paket_ID": "TL-S6-003", "Rolle": "Tech Lead", "Titel": "Sicherheit & Compliance-Verbesserung", "Dauer_Tage": 1, "Priorität": "Hoch", "Beschreibung": "Post-Launch Sicherheitsüberprüfung", "Liefergegenstände": "Security Review, Compliance-Features, Zertifizierung", "Status": "Geplant"},
    
    {"Sprint": "Sprint 6", "Paket_ID": "FE-S6-001", "Rolle": "Frontend Developer", "Titel": "User-Experience-Optimierung & A/B-Testing", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "UX basierend auf Benutzerdaten optimieren", "Liefergegenstände": "UX-Optimierung, A/B-Testing Framework, Workflow-Tuning", "Status": "Geplant"},
    {"Sprint": "Sprint 6", "Paket_ID": "FE-S6-002", "Rolle": "Frontend Developer", "Titel": "Erweiterte Features & kundenangeforderte Verbesserungen", "Dauer_Tage": 2, "Priorität": "Mittel", "Beschreibung": "Kundenangeforderte UI-Features implementieren", "Liefergegenstände": "Neue Features, Analytics-Viz, Dashboard-Anpassung", "Status": "Geplant"},
    {"Sprint": "Sprint 6", "Paket_ID": "FE-S6-003", "Rolle": "Frontend Developer", "Titel": "Mobile-App-Entwicklungsplanung", "Dauer_Tage": 1, "Priorität": "Niedrig", "Beschreibung": "Mobile App Strategie entwickeln", "Liefergegenstände": "Mobile-Strategie, Tech-Specs, Projektplan", "Status": "Geplant"},
    
    {"Sprint": "Sprint 6", "Paket_ID": "PM-S6-001", "Rolle": "Product Manager", "Titel": "Wachstumsstrategie & Marktexpansionsplanung", "Dauer_Tage": 2, "Priorität": "Kritisch", "Beschreibung": "Wachstumsstrategie für nächste Phase", "Liefergegenstände": "Wachstumsstrategie, Expansion-Plan, Metriken-Framework", "Status": "Geplant"},
    {"Sprint": "Sprint 6", "Paket_ID": "PM-S6-002", "Rolle": "Product Manager", "Titel": "Partnerschaft & Channel-Entwicklung", "Dauer_Tage": 2, "Priorität": "Hoch", "Beschreibung": "Partnerschaftsnetzwerk erweitern", "Liefergegenstände": "5+ Partner, Channel-Programm, Internationale Expansion", "Status": "Geplant"},
    {"Sprint": "Sprint 6", "Paket_ID": "PM-S6-003", "Rolle": "Product Manager", "Titel": "Produkt-Roadmap & Zukunftsplanung", "Dauer_Tage": 1, "Priorität": "Mittel", "Beschreibung": "12-Monats Produkt-Roadmap erstellen", "Liefergegenstände": "12M-Roadmap, Tech-Forschung, Team-Skalierung", "Status": "Geplant"}
]

def create_excel_file():
    """Erstellt Excel-Datei mit allen Arbeitspaketen"""
    
    # DataFrame aus Arbeitspaketen erstellen
    df = pd.DataFrame(WORK_PACKAGES)
    
    # Zusätzliche Spalten für Tracking
    df['Start_Datum'] = ''
    df['End_Datum'] = ''
    df['Tatsächliche_Dauer'] = ''
    df['Fortschritt_%'] = 0
    df['Notizen'] = ''
    df['Abhängigkeiten'] = ''
    
    # Datei speichern
    # Use config for output path if available
    if CONFIG_AVAILABLE and config:
        output_file = config.OUTPUT_DIR / 'Swiss_RAG_Arbeitspakete.xlsx'
    else:
        output_file = 'Swiss_RAG_Arbeitspakete.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Hauptsheet mit allen Arbeitspaketen
        df.to_excel(writer, sheet_name='Alle_Arbeitspakete', index=False)
        
        # Sprint-spezifische Sheets
        for sprint_info in SPRINTS:
            sprint_name = sprint_info['name'].replace(':', '').replace(' ', '_')[:30]  # Excel sheet name limit
            sprint_df = df[df['Sprint'] == sprint_info['name']]
            sprint_df.to_excel(writer, sheet_name=sprint_name, index=False)
        
        # Rollen-spezifische Sheets
        for role in ['Tech Lead', 'Frontend Developer', 'Product Manager']:
            role_name = role.replace(' ', '_')
            role_df = df[df['Rolle'] == role]
            role_df.to_excel(writer, sheet_name=role_name, index=False)
        
        # Sprint-Übersicht Sheet
        sprint_overview = []
        for sprint_info in SPRINTS:
            sprint_tasks = df[df['Sprint'] == sprint_info['name']]
            sprint_overview.append({
                'Sprint': sprint_info['name'],
                'Start_Datum': sprint_info['start'],
                'End_Datum': sprint_info['end'],
                'Anzahl_Arbeitspakete': len(sprint_tasks),
                'Gesamt_Tage': sprint_tasks['Dauer_Tage'].sum(),
                'Kritische_Pakete': len(sprint_tasks[sprint_tasks['Priorität'] == 'Kritisch']),
                'Tech_Lead_Pakete': len(sprint_tasks[sprint_tasks['Rolle'] == 'Tech Lead']),
                'Frontend_Pakete': len(sprint_tasks[sprint_tasks['Rolle'] == 'Frontend Developer']),
                'PM_Pakete': len(sprint_tasks[sprint_tasks['Rolle'] == 'Product Manager'])
            })
        
        sprint_overview_df = pd.DataFrame(sprint_overview)
        sprint_overview_df.to_excel(writer, sheet_name='Sprint_Übersicht', index=False)
        
        # Team-Workload Sheet
        team_workload = []
        for role in ['Tech Lead', 'Frontend Developer', 'Product Manager']:
            role_tasks = df[df['Rolle'] == role]
            for sprint_info in SPRINTS:
                sprint_tasks = role_tasks[role_tasks['Sprint'] == sprint_info['name']]
                team_workload.append({
                    'Rolle': role,
                    'Sprint': sprint_info['name'],
                    'Anzahl_Pakete': len(sprint_tasks),
                    'Gesamt_Tage': sprint_tasks['Dauer_Tage'].sum(),
                    'Kritische_Pakete': len(sprint_tasks[sprint_tasks['Priorität'] == 'Kritisch']),
                    'Auslastung_%': min(100, (sprint_tasks['Dauer_Tage'].sum() / 5) * 100)  # 5 Arbeitstage pro Sprint
                })
        
        team_workload_df = pd.DataFrame(team_workload)
        team_workload_df.to_excel(writer, sheet_name='Team_Auslastung', index=False)
    
    print(f"[OK] Excel-Datei erstellt: {output_file}")
    print("\nUebersicht:")
    print(f"   Gesamt Arbeitspakete: {len(df)}")
    print(f"   Gesamt Entwicklungstage: {df['Dauer_Tage'].sum()}")
    print(f"   Kritische Pakete: {len(df[df['Priorität'] == 'Kritisch'])}")
    print(f"   Sprints: {len(SPRINTS)}")
    print(f"   Team-Mitglieder: 3")
    
    print("\nSheets erstellt:")
    print("   - Alle_Arbeitspakete (Hauptliste)")
    print("   - Sprint_1 bis Sprint_6 (Pro Sprint)")
    print("   - Tech_Lead, Frontend_Developer, Product_Manager (Pro Rolle)")
    print("   - Sprint_Uebersicht (Zusammenfassung)")
    print("   - Team_Auslastung (Workload-Analyse)")
    
    return output_file

if __name__ == "__main__":
    try:
        excel_file = create_excel_file()
        print(f"\nBereit fuer die Schweizer Markteinfuehrung!")
        print(f"Datei oeffnen mit: start excel {excel_file}")
    except Exception as e:
        print(f"[ERROR] Fehler beim Erstellen der Excel-Datei: {e}")
        print("Stelle sicher, dass pandas und openpyxl installiert sind:")
        print("   pip install pandas openpyxl")