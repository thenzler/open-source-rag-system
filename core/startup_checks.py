#!/usr/bin/env python3
"""
Startup Dependency Checker for RAG System
Verifies all required components are available before starting the system
"""

import importlib.util
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Try to import configuration
try:
    from config.config import config

    CONFIG_AVAILABLE = True
except ImportError:
    config = None
    CONFIG_AVAILABLE = False


class Colors:
    """ANSI color codes for console output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


class StartupChecker:
    """Comprehensive startup dependency checker"""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.errors = []

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

    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        self.print_status("Checking Python version...")

        version = sys.version_info
        required_major, required_minor = 3, 8

        if version.major >= required_major and version.minor >= required_minor:
            self.print_status(
                f"Python {version.major}.{version.minor}.{version.micro} is compatible",
                "PASS",
            )
            return True
        else:
            self.print_status(
                f"Python {version.major}.{version.minor}.{version.micro} is too old (need >= {required_major}.{required_minor})",
                "FAIL",
            )
            self.errors.append(
                f"Upgrade Python to {required_major}.{required_minor} or higher"
            )
            return False

    def check_required_packages(self) -> bool:
        """Check if all required Python packages are installed"""
        self.print_status("Checking required Python packages...")

        required_packages = [
            ("fastapi", "FastAPI web framework"),
            ("uvicorn", "ASGI server"),
            ("sentence_transformers", "Sentence embeddings"),
            ("numpy", "Numerical computing"),
            ("requests", "HTTP library"),
            ("PyPDF2", "PDF processing"),
            ("docx", "DOCX processing"),
            ("pandas", "Data manipulation"),
            ("sklearn", "Machine learning utils"),
            ("slowapi", "Rate limiting"),
            ("multipart", "File uploads"),
        ]

        missing_packages = []

        for package, description in required_packages:
            try:
                spec = importlib.util.find_spec(package.replace("-", "_"))
                if spec is None:
                    missing_packages.append((package, description))
                else:
                    self.print_status(f"  {package} - {description}", "PASS")
            except ImportError:
                missing_packages.append((package, description))

        if missing_packages:
            self.print_status(
                f"Missing {len(missing_packages)} required packages:", "FAIL"
            )
            for package, description in missing_packages:
                self.print_status(f"  Missing: {package} - {description}", "FAIL")
            self.errors.append(
                "Install missing packages with: pip install -r simple_requirements.txt"
            )
            return False

        self.print_status("All required packages are installed", "PASS")
        return True

    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running and accessible"""
        self.print_status("Checking Ollama service...")

        # Check if Ollama is running on default port
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.print_status("Ollama service is running", "PASS")
                return True
            else:
                self.print_status(
                    f"Ollama service returned status {response.status_code}", "FAIL"
                )
                self.errors.append("Start Ollama service: ollama serve")
                return False
        except requests.exceptions.ConnectionError:
            self.print_status("Ollama service is not running", "FAIL")
            self.errors.append("Start Ollama service: ollama serve")
            return False
        except requests.exceptions.Timeout:
            self.print_status("Ollama service is not responding", "FAIL")
            self.errors.append("Check Ollama service health: ollama serve")
            return False
        except Exception as e:
            self.print_status(f"Unexpected error checking Ollama: {e}", "FAIL")
            self.errors.append("Check Ollama installation and service")
            return False

    def check_ollama_models(self) -> bool:
        """Check if required Ollama models are available"""
        self.print_status("Checking Ollama models...")

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code != 200:
                self.print_status(
                    "Cannot check models - Ollama service not available", "FAIL"
                )
                return False

            models_data = response.json()
            available_models = [
                model["name"] for model in models_data.get("models", [])
            ]

            if not available_models:
                self.print_status("No Ollama models found", "WARN")
                self.warnings += 1
                self.errors.append(
                    "Download a model: ollama pull llama2 (or your preferred model)"
                )
                return False

            self.print_status(f"Found {len(available_models)} Ollama models:", "PASS")
            for model in available_models:
                self.print_status(f"  - {model}")

            return True

        except Exception as e:
            self.print_status(f"Error checking Ollama models: {e}", "FAIL")
            self.errors.append("Check Ollama models: ollama list")
            return False

    def check_storage_directories(self) -> bool:
        """Check if required storage directories exist and are writable"""
        self.print_status("Checking storage directories...")

        if CONFIG_AVAILABLE and config:
            base_dir = config.BASE_DIR
            required_dirs = [
                config.DATA_DIR,
                config.UPLOAD_DIR,
                config.PROCESSED_DIR,
                config.BASE_DIR / "logs",
            ]
        else:
            base_dir = Path(".")
            required_dirs = [
                base_dir / "data" / "storage",
                base_dir / "data" / "storage" / "uploads",
                base_dir / "data" / "storage" / "processed",
                base_dir / "logs",
            ]

        all_good = True

        for dir_path in required_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)

                # Test write permissions
                test_file = dir_path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()

                self.print_status(f"  {dir_path} - writable", "PASS")

            except Exception as e:
                self.print_status(f"  {dir_path} - not writable: {e}", "FAIL")
                self.errors.append(f"Ensure {dir_path} is writable")
                all_good = False

        return all_good

    def check_port_availability(self) -> bool:
        """Check if required ports are available"""
        self.print_status("Checking port availability...")

        required_ports = [
            (8001, "RAG API Server"),
            (11434, "Ollama Service"),
        ]

        all_available = True

        for port, service in required_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(("localhost", port))

                    if result == 0:
                        if port == 11434:  # Ollama should be running
                            self.print_status(
                                f"  Port {port} ({service}) - in use (expected)", "PASS"
                            )
                        else:  # API port should be free
                            self.print_status(
                                f"  Port {port} ({service}) - in use (may conflict)",
                                "WARN",
                            )
                            self.warnings += 1
                    else:
                        if port == 8001:  # API port should be free
                            self.print_status(
                                f"  Port {port} ({service}) - available", "PASS"
                            )
                        else:  # Ollama should be running
                            self.print_status(
                                f"  Port {port} ({service}) - not available", "FAIL"
                            )
                            all_available = False

            except Exception as e:
                self.print_status(f"  Error checking port {port}: {e}", "FAIL")
                all_available = False

        return all_available

    def check_system_resources(self) -> bool:
        """Check basic system resources"""
        self.print_status("Checking system resources...")

        try:
            # Check available disk space
            import shutil

            total, used, free = shutil.disk_usage("C:")
            free_gb = free // (1024**3)

            if free_gb < 1:
                self.print_status(f"Low disk space: {free_gb}GB free", "WARN")
                self.warnings += 1
            else:
                self.print_status(f"Disk space: {free_gb}GB free", "PASS")

            # Check if running as administrator (Windows)
            if os.name == "nt":
                try:
                    import ctypes

                    is_admin = ctypes.windll.shell32.IsUserAnAdmin()
                    if not is_admin:
                        self.print_status("Not running as administrator", "WARN")
                        self.warnings += 1
                    else:
                        self.print_status("Running as administrator", "PASS")
                except Exception:
                    self.print_status("Cannot check admin status", "WARN")
                    self.warnings += 1

            return True

        except Exception as e:
            self.print_status(f"Error checking system resources: {e}", "WARN")
            self.warnings += 1
            return True  # Don't fail on resource checks

    def run_all_checks(self) -> Dict[str, bool]:
        """Run all startup checks and return results"""
        print(f"{Colors.BOLD}RAG System Startup Dependency Checker{Colors.END}")
        print("=" * 50)

        results = {
            "python_version": self.check_python_version(),
            "required_packages": self.check_required_packages(),
            "ollama_service": self.check_ollama_service(),
            "ollama_models": self.check_ollama_models(),
            "storage_directories": self.check_storage_directories(),
            "port_availability": self.check_port_availability(),
            "system_resources": self.check_system_resources(),
        }

        # Count results
        passed = sum(1 for result in results.values() if result)
        failed = sum(1 for result in results.values() if not result)

        print("\n" + "=" * 50)
        print(f"{Colors.BOLD}Startup Check Summary{Colors.END}")

        if failed == 0:
            print(
                f"{Colors.GREEN}[OK] All checks passed! System ready to start.{Colors.END}"
            )
            if self.warnings > 0:
                print(
                    f"{Colors.YELLOW}[WARN] {self.warnings} warnings found{Colors.END}"
                )
        else:
            print(
                f"{Colors.RED}[FAIL] {failed} checks failed. System cannot start.{Colors.END}"
            )
            if self.warnings > 0:
                print(
                    f"{Colors.YELLOW}[WARN] {self.warnings} warnings found{Colors.END}"
                )

        print(f"Passed: {passed}/{len(results)}")

        if self.errors:
            print(f"\n{Colors.RED}Required Actions:{Colors.END}")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")

        return results


def main():
    """Main function to run startup checks"""
    checker = StartupChecker()
    results = checker.run_all_checks()

    # Exit with appropriate code
    all_critical_passed = all(
        results[key]
        for key in ["python_version", "required_packages", "storage_directories"]
    )

    if all_critical_passed:
        print(
            f"\n{Colors.GREEN}[OK] Critical checks passed - system can start{Colors.END}"
        )
        sys.exit(0)
    else:
        print(
            f"\n{Colors.RED}[FAIL] Critical checks failed - system cannot start{Colors.END}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
