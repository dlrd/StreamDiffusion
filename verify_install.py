"""
Post-installation verification for StreamDiffusion-R13.
Called by install.bat to verify all dependencies are correctly installed.
"""
import sys
import os


def test_import(name, import_func):
    """Test a single import and return success/failure"""
    try:
        import_func()
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False


def main():
    print("=" * 70)
    print("StreamDiffusion Installation Check")
    print("=" * 70)
    print()

    all_ok = True

    # Test 1: CUDA
    print("[1/8] Test CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  [OK] CUDA version: {torch.version.cuda}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  [OK] Total VRAM: {vram:.1f} GB")
        else:
            print("  [ERROR] CUDA not available!")
            all_ok = False
    except ImportError:
        print("  [ERROR] PyTorch not installed!")
        all_ok = False
    print()

    # Test 2: Diffusers
    print("[2/8] Test diffusers...")
    all_ok &= test_import("diffusers", lambda: __import__("diffusers"))
    print()

    # Test 3: StreamDiffusion (local src/ folder, not pip package)
    print("[3/8] Test streamdiffusion...")
    all_ok &= test_import("src.streamdiffusion",
        lambda: __import__("src.streamdiffusion", fromlist=["StreamDiffusion"]))
    print()

    # Test 4: ControlNet
    print("[4/8] Test controlnet-aux...")
    all_ok &= test_import("controlnet-aux", lambda: __import__("controlnet_aux"))
    print()

    # Test 5: easy-dwpose
    print("[5/8] Test easy-dwpose...")
    all_ok &= test_import("easy-dwpose", lambda: __import__("easy_dwpose"))
    print()

    # Test 6: bitsandbytes
    print("[6/8] Test bitsandbytes...")
    all_ok &= test_import("bitsandbytes", lambda: __import__("bitsandbytes"))
    print()

    # Test 7: pywin32
    print("[7/8] Test pywin32 (Windows IPC)...")
    all_ok &= test_import("pywin32", lambda: __import__("win32event"))
    print()

    # Test 8: tokenizers
    print("[8/8] Test tokenizers...")
    all_ok &= test_import("tokenizers", lambda: __import__("tokenizers"))
    print()

    print("=" * 70)
    if all_ok:
        print("Installation completed successfully!")
    else:
        print("Installation incomplete - check the errors above")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
