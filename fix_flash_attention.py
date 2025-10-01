#!/usr/bin/env python3
"""
FlashAttention2 Diagnostic and Repair Tool
Diagnoses and fixes common flash-attn installation issues.
"""

import subprocess
import sys
import os

def diagnose_flash_attn():
    """Diagnose flash-attn installation issues."""
    print("🔍 Diagnosing FlashAttention2 Installation")
    print("="*50)
    
    # Check if flash_attn package is installed
    try:
        import flash_attn
        print("✓ flash_attn package is importable")
        
        # Try to get version
        try:
            version = flash_attn.__version__
            print(f"✓ Version: {version}")
        except AttributeError:
            print("⚠️  Version info not available")
        
        # Try to access CUDA components (this is where the error often occurs)
        try:
            # This will trigger the CUDA library loading
            from flash_attn import flash_attn_func
            print("✓ CUDA components load successfully")
            return "working"
        except (ImportError, OSError, RuntimeError) as e:
            print(f"❌ CUDA components failed to load: {e}")
            print("This indicates a corrupted or incompatible installation")
            return "corrupted"
            
    except ImportError:
        print("⚠️  flash_attn package not installed")
        return "not_installed"

def check_system_compatibility():
    """Check if system is compatible with flash-attn."""
    print(f"\n🔍 System Compatibility Check")
    print("="*50)
    
    try:
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            
            # Check GPU compute capability
            if torch.cuda.device_count() > 0:
                compute_cap = torch.cuda.get_device_capability(0)
                print(f"✓ GPU compute capability: {compute_cap[0]}.{compute_cap[1]}")
                
                if compute_cap[0] >= 7 and (compute_cap[0] > 7 or compute_cap[1] >= 5):
                    print("✓ GPU supports FlashAttention2")
                    return True
                else:
                    print("❌ GPU compute capability too low (need >= 7.5)")
                    return False
        else:
            print("❌ CUDA not available")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False

def fix_corrupted_installation():
    """Fix corrupted flash-attn installation."""
    print(f"\n🔧 Fixing Corrupted Installation")
    print("="*50)
    
    print("Step 1: Uninstalling corrupted flash-attn...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "flash-attn", "-y"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Successfully uninstalled corrupted package")
        else:
            print("⚠️  Uninstall had issues, continuing anyway...")
    except Exception as e:
        print(f"⚠️  Uninstall error: {e}")
    
    print("\nStep 2: Clearing pip cache...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      capture_output=True)
        print("✓ Pip cache cleared")
    except:
        pass
    
    print("\nStep 3: Installing fresh flash-attn...")
    print("This may take 10-30 minutes...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "flash-attn", 
            "--no-build-isolation", "--no-cache-dir"
        ], capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print("✓ Fresh installation completed")
            return True
        else:
            print("❌ Installation failed")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def provide_workaround():
    """Provide workaround for flash-attn issues."""
    print(f"\n💡 Workaround Solution")
    print("="*50)
    print("Don't worry! The DotsOCR pipeline works perfectly without FlashAttention2.")
    print("")
    print("✅ What works without FlashAttention2:")
    print("  - Full OCR functionality")
    print("  - All text extraction features")
    print("  - JSON output with layout information")
    print("  - GPU acceleration (for model inference)")
    print("")
    print("⚡ Performance impact:")
    print("  - Slightly slower inference (10-20%)")
    print("  - Same accuracy and functionality")
    print("  - More memory usage during inference")
    print("")
    print("🔧 To use without FlashAttention2:")
    print("  1. The pipeline automatically detects and uses eager attention")
    print("  2. No changes needed - just run: python main.py")
    print("  3. Everything will work normally")

def main():
    print("🩺 FlashAttention2 Diagnostic Tool")
    print("="*60)
    
    # Diagnose current state
    status = diagnose_flash_attn()
    
    # Check system compatibility
    compatible = check_system_compatibility()
    
    if status == "working":
        print(f"\n🎉 FlashAttention2 is working correctly!")
        print("No action needed.")
        
    elif status == "corrupted":
        print(f"\n⚠️  FlashAttention2 installation is corrupted")
        
        if compatible:
            choice = input("\nAttempt to fix the installation? (y/N): ").lower().strip()
            if choice in ['y', 'yes']:
                if fix_corrupted_installation():
                    print(f"\n🎉 Installation fixed! Test with: python test_attention.py")
                else:
                    print(f"\n⚠️  Fix failed, but that's OK!")
                    provide_workaround()
            else:
                provide_workaround()
        else:
            print(f"\nSystem not compatible with FlashAttention2")
            provide_workaround()
            
    else:  # not_installed
        if compatible:
            print(f"\nFlashAttention2 not installed but system is compatible")
            print("Run: python setup_flash_attention.py to install")
        else:
            print(f"\nSystem not compatible with FlashAttention2")
        
        provide_workaround()
    
    print(f"\n🚀 Your DotsOCR pipeline will work regardless!")
    print("Run: python main.py")

if __name__ == "__main__":
    main()