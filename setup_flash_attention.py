#!/usr/bin/env python3
"""
Flash Attention 2 Installation Helper
Provides guidance and optional installation of flash-attn for better performance.
"""

import subprocess
import sys
import torch

def check_cuda_compatibility():
    """Check if system supports FlashAttention2."""
    print("🔍 Checking CUDA Compatibility for FlashAttention2")
    print("="*50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available - FlashAttention2 requires CUDA")
        return False
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    print(f"✓ CUDA available: {cuda_version}")
    
    # Check GPU compute capability
    if torch.cuda.device_count() > 0:
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ Compute capability: {compute_capability[0]}.{compute_capability[1]}")
        
        # FlashAttention2 requires compute capability >= 7.5
        if compute_capability[0] < 7 or (compute_capability[0] == 7 and compute_capability[1] < 5):
            print("❌ GPU compute capability too low for FlashAttention2 (requires >= 7.5)")
            return False
        else:
            print("✓ GPU supports FlashAttention2")
            return True
    
    return False

def check_flash_attn_installed():
    """Check if flash-attn is already installed."""
    try:
        import flash_attn
        print("✓ flash-attn is already installed")
        return True
    except ImportError:
        print("⚠️  flash-attn not installed")
        return False

def install_flash_attn():
    """Attempt to install flash-attn."""
    print("\n📦 Installing FlashAttention2")
    print("="*50)
    print("This may take 10-30 minutes as it compiles CUDA kernels...")
    
    try:
        # Install flash-attn
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print("✅ FlashAttention2 installed successfully!")
            return True
        else:
            print("❌ FlashAttention2 installation failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out (took longer than 30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def provide_alternatives():
    """Provide alternatives if FlashAttention2 can't be installed."""
    print("\n🔧 Alternative Solutions")
    print("="*50)
    print("If FlashAttention2 installation fails, don't worry!")
    print("The DotsOCR pipeline will automatically fallback to:")
    
    print("\n✅ Eager Attention (Default)")
    print("  - Works on all systems (CPU/GPU)")
    print("  - Slightly slower but fully functional")
    print("  - No additional installation required")
    
    print("\n⚡ Performance Tips Without FlashAttention2:")
    print("  - Use GPU if available (automatically detected)")
    print("  - Use smaller batch sizes to fit in memory")
    print("  - Process PDFs one at a time for large documents")

def main():
    print("⚡ FlashAttention2 Setup Assistant")
    print("="*60)
    print("FlashAttention2 provides faster transformer inference on GPU")
    print("It's optional - the pipeline works fine without it!")
    print("="*60)
    
    # Check if already installed
    if check_flash_attn_installed():
        print("\n🎉 FlashAttention2 is ready to use!")
        return
    
    # Check compatibility
    if not check_cuda_compatibility():
        print("\n⚠️  System not compatible with FlashAttention2")
        provide_alternatives()
        return
    
    # Ask user if they want to install
    print(f"\n❓ Install FlashAttention2?")
    print("  ✅ Pros: Faster inference, better memory efficiency")
    print("  ⚠️  Cons: Long installation time, requires compilation")
    print("  ℹ️  Alternative: Pipeline works fine with eager attention")
    
    choice = input("\nInstall FlashAttention2? (y/N): ").lower().strip()
    
    if choice in ['y', 'yes']:
        if install_flash_attn():
            print("\n🎉 Setup complete! FlashAttention2 is ready to use.")
        else:
            print("\n⚠️  Installation failed, but that's OK!")
            provide_alternatives()
    else:
        print("\n👍 Skipping FlashAttention2 installation")
        provide_alternatives()
    
    print(f"\n🚀 Your DotsOCR pipeline is ready!")
    print("Run: python main.py")

if __name__ == "__main__":
    main()