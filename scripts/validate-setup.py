#!/usr/bin/env python3
"""
Validation script for DevGuard testing infrastructure.
This script validates that all testing components are properly configured.
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements"""
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, found {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_file_structure():
    """Check if all required files and directories exist"""
    required_files = [
        "pyproject.toml",
        ".pre-commit-config.yaml",
        "pytest.ini",
        "Makefile",
        "tests/conftest.py",
        "tests/utils/assertions.py",
        "tests/utils/helpers.py",
        "tests/unit/test_sample.py",
        "tests/README.md",
        ".github/workflows/ci.yml"
    ]
    
    required_dirs = [
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "tests/security",
        "tests/utils",
        "scripts"
    ]
    
    all_good = True
    
    print("\n📁 Checking file structure...")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            all_good = False
    
    return all_good

def check_pyproject_config():
    """Check pyproject.toml configuration"""
    print("\n⚙️ Checking pyproject.toml configuration...")
    
    try:
        with open("pyproject.toml") as f:
            content = f.read()
        
        checks = [
            ("pytest>=7.0.0", "pytest dependency"),
            ("pytest-cov>=4.1.0", "pytest-cov dependency"),
            ("black>=23.0.0", "black dependency"),
            ("ruff>=0.1.0", "ruff dependency"),
            ("mypy>=1.5.0", "mypy dependency"),
            ("pre-commit>=3.0.0", "pre-commit dependency"),
            ("[tool.pytest.ini_options]", "pytest configuration"),
            ("[tool.coverage.run]", "coverage configuration"),
            ("[tool.ruff]", "ruff configuration"),
            ("[tool.mypy]", "mypy configuration")
        ]
        
        all_good = True
        for check, description in checks:
            if check in content:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - Missing")
                all_good = False
        
        return all_good
        
    except FileNotFoundError:
        print("❌ pyproject.toml not found")
        return False

def check_test_configuration():
    """Check test configuration files"""
    print("\n🧪 Checking test configuration...")
    
    configs = [
        ("pytest.ini", ["testpaths = tests", "python_files = test_*.py"]),
        (".pre-commit-config.yaml", ["black", "ruff", "mypy"]),
        ("tests/conftest.py", ["pytest_plugins", "@pytest.fixture"])
    ]
    
    all_good = True
    for config_file, required_content in configs:
        if Path(config_file).exists():
            try:
                with open(config_file) as f:
                    content = f.read()
                
                file_good = True
                for required in required_content:
                    if required in content:
                        print(f"✅ {config_file} - {required}")
                    else:
                        print(f"❌ {config_file} - Missing: {required}")
                        file_good = False
                        all_good = False
                
                if file_good:
                    print(f"✅ {config_file} - Configuration valid")
                    
            except Exception as e:
                print(f"❌ {config_file} - Error reading: {e}")
                all_good = False
        else:
            print(f"❌ {config_file} - File missing")
            all_good = False
    
    return all_good

def check_scripts():
    """Check setup and utility scripts"""
    print("\n📜 Checking scripts...")
    
    scripts = [
        "scripts/setup-dev.sh",
        "scripts/run-tests.sh"
    ]
    
    all_good = True
    for script in scripts:
        if Path(script).exists():
            # Check if script is executable
            if os.access(script, os.X_OK):
                print(f"✅ {script} - Exists and executable")
            else:
                print(f"⚠️ {script} - Exists but not executable")
        else:
            print(f"❌ {script} - Missing")
            all_good = False
    
    return all_good

def check_makefile():
    """Check Makefile targets"""
    print("\n🔨 Checking Makefile...")
    
    if not Path("Makefile").exists():
        print("❌ Makefile - Missing")
        return False
    
    try:
        with open("Makefile") as f:
            content = f.read()
        
        required_targets = [
            "test:", "test-unit:", "test-integration:", 
            "lint:", "format:", "type-check:", "coverage:",
            "quality:", "pre-commit:", "clean:"
        ]
        
        all_good = True
        for target in required_targets:
            if target in content:
                print(f"✅ Makefile target: {target.rstrip(':')}")
            else:
                print(f"❌ Makefile target missing: {target.rstrip(':')}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ Makefile - Error reading: {e}")
        return False

def check_ci_configuration():
    """Check CI/CD configuration"""
    print("\n🤖 Checking CI/CD configuration...")
    
    ci_file = ".github/workflows/ci.yml"
    if not Path(ci_file).exists():
        print(f"❌ {ci_file} - Missing")
        return False
    
    try:
        with open(ci_file) as f:
            content = f.read()
        
        required_jobs = [
            "quality-checks:", "unit-tests:", "integration-tests:",
            "security-tests:", "build-and-test:"
        ]
        
        all_good = True
        for job in required_jobs:
            if job in content:
                print(f"✅ CI job: {job.rstrip(':')}")
            else:
                print(f"❌ CI job missing: {job.rstrip(':')}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ {ci_file} - Error reading: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 DevGuard Testing Infrastructure Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("File Structure", check_file_structure),
        ("PyProject Config", check_pyproject_config),
        ("Test Configuration", check_test_configuration),
        ("Scripts", check_scripts),
        ("Makefile", check_makefile),
        ("CI Configuration", check_ci_configuration)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} - Error during check: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All validation checks passed!")
        print("Your testing infrastructure is properly configured.")
        print("\nNext steps:")
        print("1. Install Python 3.10+ if not already installed")
        print("2. Run: ./scripts/setup-dev.sh")
        print("3. Run: make test")
        return True
    else:
        print(f"\n⚠️ {total - passed} validation checks failed!")
        print("Please fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)