#!/usr/bin/env python3
"""
One-Click Setup Script for RAG System
Handles complete installation and configuration
"""

import os
import sys
import subprocess
import shutil
import platform
import requests
import time
from pathlib import Path
from typing import Optional, List, Dict

class Colors:
    """ANSI color codes for console output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class RagSystemSetup:
    """Complete RAG System setup and configuration"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent.absolute()
        self.errors = []
        self.warnings = []
        
    def print_status(self, message: str, status: str = "INFO"):
        """Print formatted status message"""
        if status == "PASS":
            print(f"{Colors.GREEN}[OK]{Colors.END} {message}")
        elif status == "FAIL":
            print(f"{Colors.RED}[FAIL]{Colors.END} {message}")
        elif status == "WARN":
            print(f"{Colors.YELLOW}[WARN]{Colors.END} {message}")
        elif status == "INFO":
            print(f"{Colors.BLUE}[INFO]{Colors.END} {message}")
        else:
            print(f"  {message}")
    
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{title}{Colors.END}")
        print("=" * len(title))
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        self.print_status("Checking Python version...")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self.print_status(f"Python {version.major}.{version.minor}.{version.micro} is compatible", "PASS")
            return True
        else:
            self.print_status(f"Python {version.major}.{version.minor}.{version.micro} is too old (need >= 3.8)", "FAIL")
            self.errors.append("Please upgrade Python to 3.8 or higher")
            return False
    
    def check_pip_available(self) -> bool:
        """Check if pip is available"""
        self.print_status("Checking pip availability...")
        
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.print_status("pip is available", "PASS")
                return True
            else:
                self.print_status("pip is not available", "FAIL")
                self.errors.append("Install pip: python -m ensurepip --upgrade")
                return False
        except Exception as e:
            self.print_status(f"Error checking pip: {e}", "FAIL")
            self.errors.append("Install pip: python -m ensurepip --upgrade")
            return False
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        self.print_status("Upgrading pip...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.print_status("pip upgraded successfully", "PASS")
                return True
            else:
                self.print_status("Failed to upgrade pip", "WARN")
                self.warnings.append("pip upgrade failed, continuing anyway")
                return True  # Not critical
        except Exception as e:
            self.print_status(f"Error upgrading pip: {e}", "WARN")
            self.warnings.append("pip upgrade failed, continuing anyway")
            return True  # Not critical
    
    def install_requirements(self) -> bool:
        """Install required Python packages"""
        self.print_status("Installing Python packages...")
        
        # Essential packages without specific versions for better compatibility
        essential_packages = [
            "fastapi", "uvicorn", "sentence-transformers", "numpy", 
            "requests", "PyPDF2", "python-docx", "pandas", 
            "scikit-learn", "slowapi", "python-multipart"
        ]
        
        # Try to install packages one by one for better error handling
        failed_packages = []
        
        for package in essential_packages:
            self.print_status(f"Installing {package}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                
                if result.returncode == 0:
                    self.print_status(f"  {package} installed", "PASS")
                else:
                    self.print_status(f"  {package} failed", "WARN")
                    failed_packages.append(package)
                    self.warnings.append(f"Failed to install {package}")
                    
            except subprocess.TimeoutExpired:
                self.print_status(f"  {package} installation timed out", "WARN")
                failed_packages.append(package)
                self.warnings.append(f"Installation of {package} timed out")
                
            except Exception as e:
                self.print_status(f"  {package} error: {e}", "WARN")
                failed_packages.append(package)
                self.warnings.append(f"Error installing {package}")
        
        # Check if critical packages failed
        critical_packages = ["fastapi", "uvicorn", "requests", "numpy"]
        critical_failed = [pkg for pkg in failed_packages if pkg in critical_packages]
        
        if critical_failed:
            self.print_status(f"Critical packages failed: {critical_failed}", "FAIL")
            self.errors.append("Critical packages failed to install")
            return False
        
        if failed_packages:
            self.print_status(f"Some packages failed but system can continue: {failed_packages}", "WARN")
        else:
            self.print_status("All packages installed successfully", "PASS")
        
        return True
    
    def create_directories(self) -> bool:
        """Create required directories"""
        self.print_status("Creating project directories...")
        
        required_dirs = [
            "data/storage",
            "data/storage/uploads",
            "data/storage/processed",
            "data/logs",
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_dir / dir_name
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.print_status(f"  {dir_name} created", "PASS")
            except Exception as e:
                self.print_status(f"  {dir_name} failed: {e}", "FAIL")
                self.errors.append(f"Failed to create {dir_name}")
                return False
        
        return True
    
    def check_ollama_installation(self) -> bool:
        """Check if Ollama is installed"""
        self.print_status("Checking Ollama installation...")
        
        # Check if ollama command exists
        if shutil.which("ollama"):
            self.print_status("Ollama is installed", "PASS")
            return True
        else:
            self.print_status("Ollama is not installed", "FAIL")
            self.errors.append("Install Ollama from https://ollama.ai")
            return False
    
    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running"""
        self.print_status("Checking Ollama service...")
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.print_status("Ollama service is running", "PASS")
                return True
            else:
                self.print_status(f"Ollama service returned status {response.status_code}", "WARN")
                self.warnings.append("Start Ollama service: ollama serve")
                return False
        except requests.exceptions.ConnectionError:
            self.print_status("Ollama service is not running", "WARN")
            self.warnings.append("Start Ollama service: ollama serve")
            return False
        except Exception as e:
            self.print_status(f"Error checking Ollama service: {e}", "WARN")
            self.warnings.append("Check Ollama service: ollama serve")
            return False
    
    def suggest_ollama_model(self) -> bool:
        """Suggest downloading an Ollama model"""
        self.print_status("Checking Ollama models...")
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('models', [])
                
                if models:
                    self.print_status(f"Found {len(models)} models installed", "PASS")
                    for model in models:
                        self.print_status(f"  - {model.get('name', 'Unknown')}")
                    return True
                else:
                    self.print_status("No models found", "WARN")
                    self.warnings.append("Download a model: ollama pull mistral (recommended)")
                    return False
            else:
                self.print_status("Cannot check models - Ollama service not available", "WARN")
                return False
        except Exception as e:
            self.print_status(f"Error checking models: {e}", "WARN")
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create convenient startup scripts"""
        self.print_status("Creating startup scripts...")
        
        # Windows batch script
        if platform.system() == "Windows":
            batch_script = self.project_dir / "start_rag.bat"
            batch_content = f"""@echo off
cd /d "{self.project_dir}"
python start_simple_rag.py
pause
"""
            try:
                batch_script.write_text(batch_content)
                self.print_status("  start_rag.bat created", "PASS")
            except Exception as e:
                self.print_status(f"  start_rag.bat failed: {e}", "WARN")
                self.warnings.append("Could not create batch script")
        
        # Unix shell script
        else:
            shell_script = self.project_dir / "start_rag.sh"
            shell_content = f"""#!/bin/bash
cd "{self.project_dir}"
python3 start_simple_rag.py
"""
            try:
                shell_script.write_text(shell_content)
                shell_script.chmod(0o755)
                self.print_status("  start_rag.sh created", "PASS")
            except Exception as e:
                self.print_status(f"  start_rag.sh failed: {e}", "WARN")
                self.warnings.append("Could not create shell script")
        
        return True
    
    def run_final_test(self) -> bool:
        """Run final system test"""
        self.print_status("Running final system test...")
        
        try:
            # Import and run startup checks
            from startup_checks import StartupChecker
            
            checker = StartupChecker()
            results = checker.run_all_checks()
            
            critical_checks = ['python_version', 'required_packages', 'storage_directories']
            all_critical_passed = all(results[key] for key in critical_checks)
            
            if all_critical_passed:
                self.print_status("System test passed", "PASS")
                return True
            else:
                self.print_status("System test failed", "FAIL")
                self.errors.append("System test failed - check startup_checks output")
                return False
                
        except Exception as e:
            self.print_status(f"Error running system test: {e}", "WARN")
            self.warnings.append("Could not run system test")
            return True  # Not critical
    
    def print_setup_summary(self, success: bool):
        """Print setup summary and next steps"""
        self.print_header("Setup Summary")
        
        if success:
            print(f"{Colors.GREEN}✅ Setup completed successfully!{Colors.END}")
            print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
            print("1. Start the system:")
            if platform.system() == "Windows":
                print("   - Double-click start_rag.bat")
                print("   - Or run: python start_simple_rag.py")
            else:
                print("   - Run: ./start_rag.sh")
                print("   - Or run: python3 start_simple_rag.py")
            print("\n2. Open your browser to: http://localhost:8001")
            print("3. Upload documents and start asking questions!")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}Warnings:{Colors.END}")
                for warning in self.warnings:
                    print(f"  • {warning}")
        else:
            print(f"{Colors.RED}❌ Setup failed!{Colors.END}")
            print(f"\n{Colors.RED}Errors that need to be fixed:{Colors.END}")
            for error in self.errors:
                print(f"  • {error}")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}Warnings:{Colors.END}")
                for warning in self.warnings:
                    print(f"  • {warning}")
    
    def run_complete_setup(self) -> bool:
        """Run the complete setup process"""
        self.print_header("RAG System One-Click Setup")
        
        setup_steps = [
            ("Python Version", self.check_python_version),
            ("Pip Availability", self.check_pip_available),
            ("Pip Upgrade", self.upgrade_pip),
            ("Python Packages", self.install_requirements),
            ("Project Directories", self.create_directories),
            ("Ollama Installation", self.check_ollama_installation),
            ("Ollama Service", self.check_ollama_service),
            ("Ollama Models", self.suggest_ollama_model),
            ("Startup Scripts", self.create_startup_scripts),
            ("Final Test", self.run_final_test),
        ]
        
        # Track critical failures
        critical_failures = 0
        
        for step_name, step_func in setup_steps:
            self.print_header(f"Step: {step_name}")
            
            try:
                result = step_func()
                if not result and step_name in ["Python Version", "Pip Availability", "Python Packages", "Project Directories"]:
                    critical_failures += 1
            except Exception as e:
                self.print_status(f"Unexpected error in {step_name}: {e}", "FAIL")
                if step_name in ["Python Version", "Pip Availability", "Python Packages", "Project Directories"]:
                    critical_failures += 1
        
        # Determine overall success
        success = critical_failures == 0
        
        self.print_setup_summary(success)
        
        return success

def main():
    """Main setup function"""
    setup = RagSystemSetup()
    success = setup.run_complete_setup()
    
    if success:
        print(f"\n{Colors.GREEN}🎉 RAG System is ready to use!{Colors.END}")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}❌ Setup failed. Please fix the errors above.{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()