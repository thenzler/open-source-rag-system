# RAG System - Projekt Roadmap & TODO Liste

## 🎯 Aktueller Status
- ✅ System startet erfolgreich (Startup-Bug behoben)
- ✅ Grundfunktionalität vorhanden
- ⚠️ Qualität/Speed verbesserungswürdig (nach Server-Migration)

---

## 🔥 KRITISCHE PRIORITÄT (Jetzt - Week 1)

### KAN-5: Projekt zum laufen bringen (In Progress - Emre Sen)
- ✅ **Startup-Bug behoben** - System startet erfolgreich
- 🔄 **Server Migration** - Auf produktionsfähigen Server wechseln
- 🔄 **Basic Testing** - Grundfunktionen validieren

### KAN-76: Setting Up Project Server with GPU (In Progress - Thomas Henzler)
- 🔄 **Server Setup** - GPU Server für bessere Performance
- 🔄 **Ollama Installation** - Bessere Modelle als tinyllama
- 🔄 **Performance Baseline** - Geschwindigkeit messen

### KAN-6: Git Projekt richtig einrichten (In Progress - Thomas Henzler)
- ✅ **Repository Organisation** - Code ist strukturiert
- 🔄 **CI/CD Setup** - GitHub Actions Pipeline
- 🔄 **Branch Strategy** - Main/Dev branches

---

## 🏗️ ARCHITEKTUR & CODE (Week 2-4)

### KAN-9: Code-Refactoring & Architektur (Epic - To Do)

#### KAN-10: Monolithische simple_api.py in Module aufteilen (In Progress)
- ✅ **Bereits erledigt** - System ist bereits modular mit core/ Struktur
- ✅ **Router-basierte Architektur** - Getrennte Endpoints
- ✅ **Service Layer** - Business Logic separiert

#### KAN-14: Service Layer von API Endpoints trennen (In Progress)
- ✅ **Bereits implementiert** - services/ Verzeichnis vorhanden
- ✅ **Clean Architecture** - Repository Pattern aktiv

#### KAN-13: Dependency Injection Framework einführen (In Progress)
- ✅ **Bereits implementiert** - DI Container vorhanden (core/di/)
- ✅ **Service Configuration** - Automatische Initialisierung

#### KAN-12: Repository Pattern für Datenzugriff implementieren (In Progress)
- ✅ **Bereits implementiert** - repositories/ mit Interface Pattern
- ✅ **SQLite/Vector Storage** - Abstrahierte Datenschicht

#### KAN-11: Hardcoded Pfade durch Umgebungsvariablen ersetzen (In Progress)
- 🔄 **Config System** - Bereits vorhanden, aber .env.example fehlt
- 🔄 **Environment Variables** - Dokumentation vervollständigen

---

## 🔒 SICHERHEIT (Week 3-6)

### KAN-20: Sicherheit (Epic - To Do)

#### **Höchste Priorität:**
- 🔄 **KAN-21: Path Traversal Vulnerabilities beheben**
- 🔄 **KAN-22: CSRF Protection implementieren**
- 🔄 **KAN-23: Security Headers (CSP, HSTS, etc.)**

#### **Mittlere Priorität:**
- 🔄 **KAN-24: Document ID Randomisierung**
- 🔄 **KAN-25: Encryption at Rest für Dokumente**
- 🔄 **KAN-26: MFA Support hinzufügen**

---

## 💾 DATENBANK & PERSISTENZ (Week 4-6)

### KAN-15: Datenbank & Persistenz (Epic - To Do)

#### **Entscheidung erforderlich:**
- 🤔 **KAN-16: PostgreSQL für Metadaten** - SQLite vs PostgreSQL?
- 🤔 **KAN-17: Redis für Caching** - Notwendig oder erst später?
- 🔄 **KAN-18: Connection Pooling implementieren**
- 🔄 **KAN-19: Backup & Recovery Strategie**

---

## ⚡ PERFORMANCE & SKALIERUNG (Week 6-8)

### KAN-27: Asynchrone Verarbeitung (Epic)
- 🔄 **KAN-28: Async Document Processing Queue**
- 🔄 **KAN-29: Background Job Management**
- 🔄 **KAN-30: Progress Tracking für lange Operationen**

### KAN-40: Skalierung (Epic)
- 🔄 **KAN-41: Kubernetes Deployment**
- 🔄 **KAN-42: Horizontal Scaling Support**
- 🔄 **KAN-43: Load Balancing Configuration**
- 🔄 **KAN-44: S3/MinIO für Document Storage**

---

## 📊 MONITORING (Week 7-9)

### KAN-31: Monitoring & Observability (Epic)
- 🔄 **KAN-32: Prometheus Metrics Integration**
- 🔄 **KAN-33: Grafana Dashboards erstellen**
- 🔄 **KAN-34: Performance Monitoring**

---

## 🏢 ENTERPRISE FEATURES (Week 8-12)

### KAN-35: Enterprise Features (Epic)
- 🔄 **KAN-36: Multi-Tenancy Support** ⭐ **VERKAUFSKRITISCH**
- 🔄 **KAN-37: SSO Integration (SAML/OIDC)**
- 🔄 **KAN-38: Audit Logging (SOC2/GDPR compliant)**
- 🔄 **KAN-39: Data Retention Policies**

---

## 📝 RECHTLICHES & COMPLIANCE (Parallel)

### KAN-51: Compliance & Rechtliches (Epic)
- 🔄 **KAN-52: DSGVO/DSG Compliance-Prüfung**
- 🔄 **KAN-53: Schweizer Datenschutzrichtlinien implementieren**
- 🔄 **KAN-54: Terms of Service & Privacy Policy erstellen**
- 🔄 **KAN-55: Software-Lizenzmodell definieren**

---

## 💰 BUSINESS & MONETARISIERUNG (Parallel - Marek)

### KAN-1: Effiketiven Name finden (To Do - Marek)
### KAN-2: Potenzielle Firma finden (To Do - Marek)
### KAN-3: Webseite erstellen (To Do)
### KAN-4: Business Plan fertig/versenden (✅ Done)

### KAN-63: Monetarisierung & Business Model (Epic)
- 🔄 **KAN-64: Subscription-Tiers definieren**
- 🔄 **KAN-65: Usage-Based Pricing Model entwickeln**
- 🔄 **KAN-66: Billing & Payment Integration**
- 🔄 **KAN-67: Upselling-Strategie**

### KAN-45: Marktanalyse & Positionierung (Epic)
- 🔄 **KAN-46: Wettbewerbsanalyse**
- 🔄 **KAN-47: Zielgruppen-Definition**
- 🔄 **KAN-48: USP-Definition für Schweizer Markt**

---

## 📈 MARKETING & VERTRIEB (Month 3-6)

### KAN-59: Vertrieb & Partnerschaften (Epic)
### KAN-56: Kunden-Onboarding & Success (Epic)
### KAN-68: Marketing & Brand (Epic)
### KAN-71: Produkt-Roadmap & Innovation (Epic)

---

## 🎯 SOFORTIGE ENTSCHEIDUNGEN ERFORDERLICH

### 1. **Server-Architektur**
**Optionen:**
- A) Erstmal bei aktuellem Setup bleiben, nur GPU Server hinzufügen
- B) Komplette Migration auf Cloud (AWS/Azure)
- C) Hybrid: On-Premise GPU + Cloud Services

### 2. **Datenbank-Strategie**
**Optionen:**
- A) SQLite behalten für MVP, später migrieren
- B) Sofort auf PostgreSQL wechseln
- C) Multi-DB Support (SQLite für kleine, PostgreSQL für Enterprise)

### 3. **Authentication-Priorität**
**Optionen:**
- A) Einfache API Keys erst implementieren
- B) Vollständiges JWT + Multi-Tenancy System
- C) SSO direkt für Enterprise-Kunden

### 4. **MVP vs Enterprise**
**Optionen:**
- A) Fokus auf funktionierende Basis, dann Enterprise Features
- B) Parallel entwickeln (Basis + Multi-Tenancy)
- C) Enterprise-First Ansatz

---

## 📊 NÄCHSTE SCHRITTE (Diese Woche)

1. **Server Migration abschließen** (KAN-76)
2. **Performance Baseline erstellen**
3. **Security Vulnerabilities beheben** (KAN-21, KAN-22)
4. **Multi-Tenancy Architektur entscheiden** (KAN-36)
5. **CI/CD Pipeline setup** (KAN-6)

---

**Welche Entscheidungen soll ich als erstes angehen? Welche Option bevorzugen Sie für Server-Architektur und Datenbank-Strategie?**