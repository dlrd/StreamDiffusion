#!/usr/bin/env python
"""
Setup script for StreamDiffusion-R15 virtual environment
This script copies required binary dependencies into the .venv after pip installation
"""
import os
import sys
import shutil
from pathlib import Path


def print_step(step_num, message):
    """Print a formatted step message"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {message}")
    print(f"{'='*60}")


def check_venv():
    """Check if we're running inside a virtual environment"""
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

    if not in_venv:
        print("Warning: Not running inside a virtual environment!")
        print("   This script should be run from the activated .venv")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print("[OK] Running inside virtual environment")

    return in_venv


def find_cuda_bin():
    """Find CUDA Toolkit bin directory using environment variables or standard paths"""
    # 1. Try CUDA_PATH environment variable (set by CUDA installer)
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.exists(cuda_bin):
            print(f"[OK] Found CUDA via CUDA_PATH: {cuda_bin}")
            return cuda_bin

    # 2. Try CUDA_HOME environment variable
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        cuda_bin = os.path.join(cuda_home, "bin")
        if os.path.exists(cuda_bin):
            print(f"[OK] Found CUDA via CUDA_HOME: {cuda_bin}")
            return cuda_bin

    # 3. Fallback: scan standard Windows install paths
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(base):
        # List versions and pick the highest
        versions = sorted(
            [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
            reverse=True
        )
        for ver in versions:
            cuda_bin = os.path.join(base, ver, "bin")
            if os.path.exists(cuda_bin):
                print(f"[OK] Found CUDA via scan: {cuda_bin}")
                return cuda_bin

    return None


def copy_cuda_tools():
    """Copy CUDA tools (cuobjdump, nvdisasm) to Triton directory"""
    print_step(1, "Copying CUDA Tools to Triton")

    cuda_bin = find_cuda_bin()
    if not cuda_bin:
        print("[ERROR] CUDA Toolkit not found!")
        print("   Please install CUDA Toolkit 12.1+ from:")
        print("   https://developer.nvidia.com/cuda-downloads")
        return False

    # Check for required tools
    cuobjdump_src = os.path.join(cuda_bin, "cuobjdump.exe")
    nvdisasm_src = os.path.join(cuda_bin, "nvdisasm.exe")

    if not os.path.exists(cuobjdump_src):
        print(f"[ERROR] cuobjdump.exe not found in {cuda_bin}")
        return False

    if not os.path.exists(nvdisasm_src):
        print(f"[ERROR] nvdisasm.exe not found in {cuda_bin}")
        return False

    print(f"[OK] Found cuobjdump.exe")
    print(f"[OK] Found nvdisasm.exe")

    # Find Triton directory in .venv
    venv_root = Path(sys.prefix)
    triton_bin = venv_root / "Lib" / "site-packages" / "triton" / "backends" / "nvidia" / "bin"

    if not triton_bin.exists():
        print(f"[ERROR] Triton directory not found: {triton_bin}")
        print("   Please install requirements.txt first:")
        print("   pip install -r requirements.txt")
        return False

    print(f"[OK] Found Triton directory: {triton_bin}")

    # Copy files
    try:
        cuobjdump_dst = triton_bin / "cuobjdump.exe"
        nvdisasm_dst = triton_bin / "nvdisasm.exe"

        shutil.copy2(cuobjdump_src, cuobjdump_dst)
        print(f"[OK] Copied cuobjdump.exe -> {cuobjdump_dst}")

        shutil.copy2(nvdisasm_src, nvdisasm_dst)
        print(f"[OK] Copied nvdisasm.exe -> {nvdisasm_dst}")

        return True
    except Exception as e:
        print(f"[ERROR] copying files: {e}")
        return False


def download_python_headers(dest_dir):
    """Download Python development headers from NuGet when not available locally"""
    import urllib.request
    import zipfile
    import tempfile

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    url = f"https://www.nuget.org/api/v2/package/python/{version}"

    print(f"[INFO] Headers not available locally")
    print(f"[INFO] Downloading from NuGet (Python {version})...")

    tmp_path = os.path.join(tempfile.gettempdir(), f"python-{version}.nupkg")
    try:
        urllib.request.urlretrieve(url, tmp_path)

        # Extract only tools/include/ from the NuGet package
        extracted = 0
        with zipfile.ZipFile(tmp_path, 'r') as zf:
            for member in zf.namelist():
                if member.startswith('tools/include/') and not member.endswith('/'):
                    rel_path = member[len('tools/include/'):]
                    target = os.path.join(str(dest_dir), rel_path)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src_f, open(target, 'wb') as dst_f:
                        dst_f.write(src_f.read())
                    extracted += 1

        os.unlink(tmp_path)

        if extracted > 0:
            print(f"[OK] {extracted} headers downloaded to {dest_dir}")
            return True
        else:
            print(f"[ERROR] No header found in the NuGet package")
            return False

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False


def copy_python_headers():
    """Copy Python development headers to .venv for torch.compile() support"""
    print_step(2, "Installing Python Development Headers")

    venv_root = Path(sys.prefix)
    venv_include = venv_root / "Scripts" / "Include"

    # Check if headers already exist in the venv
    if (venv_include / "Python.h").exists():
        print(f"[OK] Python headers already present in {venv_include}")
        return True

    # Try local headers first (full Python installation)
    python_root = sys.base_prefix
    python_include = os.path.join(python_root, "include")

    if os.path.exists(python_include) and os.path.exists(os.path.join(python_include, "Python.h")):
        print(f"[OK] Found local Python headers: {python_include}")
        try:
            if venv_include.exists():
                shutil.rmtree(venv_include)
            shutil.copytree(python_include, venv_include)
            print(f"[OK] Copied Python headers -> {venv_include}")
            return True
        except Exception as e:
            print(f"[ERROR] copying headers: {e}")

    # Local headers not available (embedded Python) -> download from NuGet
    print(f"[INFO] Embedded Python detected (no local headers)")
    try:
        if venv_include.exists():
            shutil.rmtree(venv_include)
        os.makedirs(venv_include, exist_ok=True)
        success = download_python_headers(venv_include)
        if success:
            # Verify critical files
            if (venv_include / "Python.h").exists():
                print(f"[OK] Verified Python.h")
            else:
                print(f"[WARNING] Python.h not found after download")
                return False
            return True
        return False
    except Exception as e:
        print(f"[ERROR] installing headers: {e}")
        return False


def verify_setup():
    """Verify that all dependencies are properly set up"""
    print_step(3, "Verifying Setup")

    success = True

    # Check Triton CUDA tools
    venv_root = Path(sys.prefix)
    triton_bin = venv_root / "Lib" / "site-packages" / "triton" / "backends" / "nvidia" / "bin"

    cuobjdump = triton_bin / "cuobjdump.exe"
    nvdisasm = triton_bin / "nvdisasm.exe"

    if cuobjdump.exists():
        print(f"[OK] cuobjdump.exe present in Triton")
    else:
        print(f"[ERROR] cuobjdump.exe NOT found in Triton")
        success = False

    if nvdisasm.exists():
        print(f"[OK] nvdisasm.exe present in Triton")
    else:
        print(f"[ERROR] nvdisasm.exe NOT found in Triton")
        success = False

    # Check Python headers
    venv_include = venv_root / "Scripts" / "Include"
    python_h = venv_include / "Python.h"

    if python_h.exists():
        print(f"[OK] Python.h present in venv")
    else:
        print(f"[WARNING] Python.h NOT found (torch.compile() may not work)")

    return success


def main():
    print("")
    print("============================================================")
    print("  StreamDiffusion-R15 Virtual Environment Setup")
    print("  Automated Binary Dependencies Installer")
    print("============================================================")
    print("")

    # Check virtual environment
    check_venv()

    # Copy CUDA tools
    cuda_success = copy_cuda_tools()

    # Copy Python headers
    headers_success = copy_python_headers()

    # Verify setup
    verify_success = verify_setup()

    # Summary
    print_step(4, "Setup Summary")

    if cuda_success and verify_success:
        print("[OK] CUDA tools successfully installed")
        print("  -> Triton warnings will no longer appear")
    else:
        print("[ERROR] CUDA tools installation failed")
        print("  -> You may see Triton warnings during execution")

    if headers_success:
        print("[OK] Python headers successfully installed")
        print("  -> torch.compile() will work in non-TensorRT mode")
    else:
        print("[WARNING] Python headers not installed")
        print("  -> torch.compile() may fail, but TensorRT will work fine")

    print("\n" + "="*60)

    if cuda_success and headers_success and verify_success:
        print("Setup completed successfully!")
        return 0
    elif cuda_success and verify_success:
        print("Setup completed with warnings")
        print("\nStreamDiffusion will work, but torch.compile() may not be available.")
        print("TensorRT acceleration will work normally.")
        return 0
    else:
        print("Setup completed with errors")
        print("\nPlease resolve the errors above before running StreamDiffusion.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
