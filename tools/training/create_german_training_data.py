#!/usr/bin/env python3
"""
Create German/Swiss German Training Data for Arlesheim Municipal Assistant
Generates realistic municipal use cases in German and Swiss German
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import random
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class GermanTrainingDataGenerator:
    """Generate German/Swiss German training data for Arlesheim"""
    
    def __init__(self):
        self.swiss_german_phrases = {
            "greetings": ["Grüezi", "Hallo", "Guete Tag", "Salü"],
            "thank_you": ["Merci vilmal", "Danke vilmal", "Herzliche Dank"],
            "goodbye": ["Uf Widerluege", "Tschüss", "Bis spöter"],
            "exclamations": ["Gopferdeckel", "Herrjeh", "Meinsch würkli?"],
            "particles": ["scho", "halt", "eifach", "gäll", "öppe"]
        }
        
        # Real Arlesheim municipal data from our scraping
        self.arlesheim_data = {
            "office_hours": {
                "standard": "Montag bis Freitag: 08:00-12:00 und 14:00-17:00",
                "thursday": "Donnerstag: 08:00-12:00 und 14:00-18:00",
                "phone": "061 705 20 20",
                "email": "info@arlesheim.ch"
            },
            "services": [
                "Baugesuche und Baubewilligungen",
                "Zivilstandsamt (Heirat, Geburt, Tod)",
                "Einwohnermeldeamt",
                "Steuern und Abgaben",
                "Sozialhilfe",
                "Schulverwaltung",
                "Kehrichtabfuhr",
                "Wasserversorgung",
                "Friedhofsverwaltung",
                "Vermietungen (Säle, Plätze)"
            ],
            "departments": {
                "Gemeindekanzlei": "Allgemeine Verwaltung, Einwohnermeldeamt",
                "Bauamt": "Baugesuche, Baubewilligungen, Planungen",
                "Finanzamt": "Steuern, Gebühren, Rechnungswesen",
                "Sozialamt": "Sozialhilfe, Vormundschaft",
                "Schulverwaltung": "Primarschule, Sekundarschule",
                "Werkhof": "Unterhalt, Kehrichtabfuhr, Winterdienst"
            },
            "locations": {
                "Gemeindeverwaltung": "Dorfstrasse 46, 4144 Arlesheim",
                "Primarschule": "Schulstrasse 12, 4144 Arlesheim",
                "Sekundarschule": "Schulstrasse 20, 4144 Arlesheim",
                "Friedhof": "Friedhofweg, 4144 Arlesheim"
            }
        }
    
    def generate_municipal_qa_pairs(self) -> List[Dict]:
        """Generate realistic German/Swiss German Q&A pairs"""
        training_data = []
        
        # 1. Office Hours & Contact Information
        contact_questions = [
            ("Wann isch d'Gmeindverwautig offe?", "hochdeutsch"),
            ("Weli sind d'Öffnigszyte vo de Verwautig?", "schweizerdeutsch"),
            ("Wie sind die Öffnungszeiten der Gemeindeverwaltung?", "hochdeutsch"),
            ("Wann chan i uf d'Gmeind cho?", "schweizerdeutsch"),
            ("Kann ich am Donnerstag länger auf die Verwaltung?", "hochdeutsch"),
            ("Wie lautet die Telefonnummer der Gemeinde?", "hochdeutsch"),
            ("Weles isch d'Telefonnummer vo Arlesheim?", "schweizerdeutsch"),
            ("Wo chan i d'Gmeindverwautig erreiche?", "schweizerdeutsch")
        ]
        
        for question, dialect in contact_questions:
            if dialect == "schweizerdeutsch":
                answer = f"Grüezi! D'Gmeindverwautig Arlesheim isch {self.arlesheim_data['office_hours']['standard']} offe. Am Donnerstag bis 18:00. Telefon: {self.arlesheim_data['office_hours']['phone']}"
            else:
                answer = f"Die Gemeindeverwaltung Arlesheim ist {self.arlesheim_data['office_hours']['standard']} geöffnet. Donnerstags bis 18:00 Uhr. Telefon: {self.arlesheim_data['office_hours']['phone']}"
            
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "category": "contact",
                "dialect": dialect
            })
        
        # 2. Building Permits & Construction
        building_questions = [
            ("Wie chan i es Baugsuch stelle?", "schweizerdeutsch"),
            ("Was bruuchi für es Baugsuch?", "schweizerdeutsch"),
            ("Wie beantrage ich eine Baubewilligung?", "hochdeutsch"),
            ("Wo muss ich mein Baugesuch einreichen?", "hochdeutsch"),
            ("Wie lang dauert es Baugsuch-Verfahre?", "schweizerdeutsch"),
            ("Wieviel kostet eine Baubewilligung?", "hochdeutsch")
        ]
        
        for question, dialect in building_questions:
            if dialect == "schweizerdeutsch":
                answer = "Für es Baugsuch müend Sie die Plän und Unterlagen bim Bauamt vo Arlesheim yriche. Sie chönd die Formulare uf www.arlesheim.ch ablade oder bim Bauamt hole. S'Verfahre dauert je nach Projekt 2-6 Woche."
            else:
                answer = "Für eine Baubewilligung müssen Sie die Pläne und Unterlagen beim Bauamt Arlesheim einreichen. Die Formulare können Sie auf www.arlesheim.ch herunterladen oder beim Bauamt abholen. Das Verfahren dauert je nach Projekt 2-6 Wochen."
            
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "category": "building",
                "dialect": dialect
            })
        
        # 3. Civil Registry (Zivilstandsamt)
        civil_questions = [
            ("Wo chan i hürate z'Arlesheim?", "schweizerdeutsch"),
            ("Wie melde ich eine Geburt an?", "hochdeutsch"),
            ("Was bruuchi für d'Hochzyt?", "schweizerdeutsch"),
            ("Wo bekomme ich einen Geburtsschein?", "hochdeutsch"),
            ("Wie meld i es Todesfau?", "schweizerdeutsch"),
            ("Wann isch s'Zivilstandsamt offe?", "schweizerdeutsch")
        ]
        
        for question, dialect in civil_questions:
            if dialect == "schweizerdeutsch":
                answer = "S'Zivilstandsamt Arlesheim isch für Hürate, Geburte und Todesfäu zueständig. Sie chönd Termin vereinbare unter 061 705 20 20. Für Hochzyte bruuche Sie d'Unterlagen vo beide Partner."
            else:
                answer = "Das Zivilstandsamt Arlesheim ist für Heiraten, Geburten und Todesfälle zuständig. Sie können einen Termin vereinbaren unter 061 705 20 20. Für Hochzeiten benötigen Sie die Unterlagen beider Partner."
            
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "category": "civil_registry",
                "dialect": dialect
            })
        
        # 4. Taxes & Finances
        tax_questions = [
            ("Wann mues i d'Stüüre zahle?", "schweizerdeutsch"),
            ("Wie hoch sind die Steuern in Arlesheim?", "hochdeutsch"),
            ("Wo chan i mini Stüürerklärig abgeh?", "schweizerdeutsch"),
            ("Wer hilft mir bei der Steuererklärung?", "hochdeutsch"),
            ("Bis wann mues d'Stüürerklärig ygreicht si?", "schweizerdeutsch"),
            ("Kann ich die Steuern in Raten zahlen?", "hochdeutsch")
        ]
        
        for question, dialect in tax_questions:
            if dialect == "schweizerdeutsch":
                answer = "D'Stüürerklärig mues bis 31. März ygreicht werde. Sie chönd die online oder bim Finanzamt abgeh. Bi Frage hälft Ihne s'Finanzamt Arlesheim under 061 705 20 20."
            else:
                answer = "Die Steuererklärung muss bis 31. März eingereicht werden. Sie können diese online oder beim Finanzamt abgeben. Bei Fragen hilft Ihnen das Finanzamt Arlesheim unter 061 705 20 20."
            
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "category": "taxes",
                "dialect": dialect
            })
        
        # 5. Waste Management
        waste_questions = [
            ("Wann wird de Kehricht abgholt?", "schweizerdeutsch"),
            ("Wo kann ich Sperrmüll entsorgen?", "hochdeutsch"),
            ("Wann isch d'Kehrichtabfuhr?", "schweizerdeutsch"),
            ("Wie entsorge ich Grünabfälle?", "hochdeutsch"),
            ("Wo git's Kehrichtsäck?", "schweizerdeutsch"),
            ("Wann wird das Altpapier geholt?", "hochdeutsch")
        ]
        
        for question, dialect in waste_questions:
            if dialect == "schweizerdeutsch":
                answer = "D'Kehrichtabfuhr isch jede Mäntig früeh. Sperrmüll chönd Sie bim Werkhof amelde. Grüeabfäu wärde am erste Mäntig im Monet abgholt. Kehrichtsäck git's im Gmeindhuus."
            else:
                answer = "Die Kehrichtabfuhr ist jeden Montagmorgen. Sperrmüll können Sie beim Werkhof anmelden. Grünabfälle werden am ersten Montag im Monat abgeholt. Kehrichtsäcke gibt's im Gemeindehaus."
            
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "category": "waste",
                "dialect": dialect
            })
        
        # 6. Schools
        school_questions = [
            ("Wann mues i mis Chind für d'Schuel amelde?", "schweizerdeutsch"),
            ("Wie melde ich mein Kind für die Schule an?", "hochdeutsch"),
            ("Wo isch d'Primarschuel?", "schweizerdeutsch"),
            ("Wann sind die Schulferien?", "hochdeutsch"),
            ("Wie lang dauert d'Mittagspause?", "schweizerdeutsch"),
            ("Gibt es einen Mittagstisch?", "hochdeutsch")
        ]
        
        for question, dialect in school_questions:
            if dialect == "schweizerdeutsch":
                answer = "D'Primarschuel isch a de Schulstrasse 12. Für d'Aamäldig wände Sie sich a d'Schulverwautig. Es git es Mittagsangebot für d'Chind. Telefon: 061 705 20 20"
            else:
                answer = "Die Primarschule ist an der Schulstrasse 12. Für die Anmeldung wenden Sie sich an die Schulverwaltung. Es gibt ein Mittagsangebot für die Kinder. Telefon: 061 705 20 20"
            
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "category": "schools",
                "dialect": dialect
            })
        
        # 7. Events & Culture
        event_questions = [
            ("Was lauft z'Arlesheim?", "schweizerdeutsch"),
            ("Welche Veranstaltungen gibt es?", "hochdeutsch"),
            ("Wann isch s'nächste Fest?", "schweizerdeutsch"),
            ("Wo finde ich den Veranstaltungskalender?", "hochdeutsch"),
            ("Git's es Dorffest?", "schweizerdeutsch"),
            ("Kann ich einen Saal mieten?", "hochdeutsch")
        ]
        
        for question, dialect in event_questions:
            if dialect == "schweizerdeutsch":
                answer = "Alles über Verastaltige finde Sie uf www.arlesheim.ch. Es git regelmässig Feste und Aläss im Dorf. Für Saal-Vermietige wände Sie sich a d'Gmeindverwautig."
            else:
                answer = "Alle Veranstaltungen finden Sie auf www.arlesheim.ch. Es gibt regelmässig Feste und Anlässe im Dorf. Für Saal-Vermietungen wenden Sie sich an die Gemeindeverwaltung."
            
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "category": "events",
                "dialect": dialect
            })
        
        # 8. Mixed Swiss German expressions
        mixed_questions = [
            ("Gopferdeckel, wo isch denn s'WC?", "schweizerdeutsch"),
            ("Meinsch würkli, das gaht so schnäu?", "schweizerdeutsch"),
            ("Isch das scho rächt so?", "schweizerdeutsch"),
            ("Chönd Sie mir eifach säge, wo's lang gaht?", "schweizerdeutsch")
        ]
        
        for question, dialect in mixed_questions:
            answer = "Grüezi! Gern hälf i Ihne wyter. Für spezifischi Frage chönd Sie sich a d'Gmeindverwautig Arlesheim wände - mir sind da für Sie da!"
            
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "category": "general",
                "dialect": dialect
            })
        
        return training_data
    
    def create_instruction_tuning_format(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Convert to instruction tuning format for fine-tuning"""
        formatted_data = []
        
        for pair in qa_pairs:
            # Add system context for municipal assistant
            system_msg = """Du bist ein hilfsbreiter Assistent für die Gemeinde Arlesheim. Du verstehst sowohl Hochdeutsch als auch Schweizerdeutsch und antwortest in der passenden Sprache. Du kennst alle Dienstleistungen, Öffnungszeiten und Verfahren der Gemeinde Arlesheim."""
            
            formatted_data.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": pair["instruction"]},
                    {"role": "assistant", "content": pair["output"]}
                ],
                "category": pair["category"],
                "dialect": pair["dialect"]
            })
        
        return formatted_data
    
    def save_training_data(self, output_dir: str = "training_data/arlesheim_german"):
        """Generate and save German/Swiss German training data"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate Q&A pairs
        qa_pairs = self.generate_municipal_qa_pairs()
        
        # Create instruction tuning format
        instruction_data = self.create_instruction_tuning_format(qa_pairs)
        
        # Save in multiple formats
        
        # 1. JSONL format for fine-tuning
        jsonl_file = os.path.join(output_dir, "arlesheim_german_training.jsonl")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in instruction_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # 2. JSON format
        json_file = os.path.join(output_dir, "arlesheim_german_training.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(instruction_data, f, ensure_ascii=False, indent=2)
        
        # 3. Simple format for Ollama
        ollama_format = []
        for pair in qa_pairs:
            ollama_format.append({
                "prompt": pair["instruction"],
                "response": pair["output"]
            })
        
        ollama_file = os.path.join(output_dir, "arlesheim_ollama_format.json")
        with open(ollama_file, 'w', encoding='utf-8') as f:
            json.dump(ollama_format, f, ensure_ascii=False, indent=2)
        
        # 4. Statistics
        stats = {
            "total_examples": len(qa_pairs),
            "categories": {},
            "dialects": {}
        }
        
        for pair in qa_pairs:
            stats["categories"][pair["category"]] = stats["categories"].get(pair["category"], 0) + 1
            stats["dialects"][pair["dialect"]] = stats["dialects"].get(pair["dialect"], 0) + 1
        
        stats_file = os.path.join(output_dir, "training_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Created German/Swiss German training data:")
        print(f"   Output directory: {output_dir}")
        print(f"   Total examples: {len(qa_pairs)}")
        print(f"   Dialects: {list(stats['dialects'].keys())}")
        print(f"   Categories: {list(stats['categories'].keys())}")
        print(f"   Files created:")
        print(f"      - {jsonl_file}")
        print(f"      - {json_file}")
        print(f"      - {ollama_file}")
        print(f"      - {stats_file}")
        
        return output_dir, len(qa_pairs)

def main():
    """Generate German/Swiss German training data for Arlesheim"""
    generator = GermanTrainingDataGenerator()
    output_dir, num_examples = generator.save_training_data()
    
    print(f"\nGerman/Swiss German training data ready for fine-tuning!")
    print(f"Use {output_dir}/arlesheim_german_training.jsonl for fine-tuning")

if __name__ == "__main__":
    main()