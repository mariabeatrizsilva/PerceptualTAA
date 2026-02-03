"""
Check and install required packages for Phase 2.
"""
import subprocess
import sys

REQUIRED_PACKAGES = {
    'opencv-python': 'cv2',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'tqdm': 'tqdm',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn'
}

def check_package(package_name, import_name):
    """Check if a package is installed."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def main():
    print("="*80)
    print("CHECKING REQUIRED PACKAGES")
    print("="*80)
    
    missing_packages = []
    
    for package_name, import_name in REQUIRED_PACKAGES.items():
        if check_package(package_name, import_name):
            print(f"✓ {package_name} is installed")
        else:
            print(f"✗ {package_name} is NOT installed")
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n" + "="*80)
        print("MISSING PACKAGES")
        print("="*80)
        print(f"The following packages need to be installed: {', '.join(missing_packages)}")
        
        response = input("\nWould you like to install them now? (y/n): ")
        
        if response.lower() == 'y':
            for package in missing_packages:
                try:
                    install_package(package)
                    print(f"✓ Successfully installed {package}")
                except Exception as e:
                    print(f"✗ Failed to install {package}: {e}")
        else:
            print("\nPlease install manually using:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    print("\n" + "="*80)
    print("ALL PACKAGES READY!")
    print("="*80)
    print("You can now run: python extract_features.py")
    return True

if __name__ == "__main__":
    main()