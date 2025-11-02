#!/usr/bin/env python3
"""
CONFLUENCE External Tools Configuration

This module defines external tool configurations required by CONFLUENCE,
including repositories, build instructions, and validation criteria.

Tools include:
- SUNDIALS: Differential equation solver library
- SUMMA: Hydrological model with SUNDIALS integration
- mizuRoute: River network routing model
- FUSE: Framework for Understanding Structural Errors
- TauDEM: Terrain Analysis Using Digital Elevation Models
- GIStool: Geospatial data extraction tool
- Datatool: Meteorological data processing tool
- NGEN: NextGen National Water Model Framework
- NGIAB: NextGen In A Box deployment system
"""

from typing import Dict, Any


def get_common_build_environment() -> str:
    """
    Get common build environment setup used across multiple tools.
    
    Returns:
        Shell script snippet for environment configuration
    """
    return r'''
set -e
# Compiler: force short name to satisfy Makefile sanity checks
export FC="${FC:-gfortran}"
export FC_EXE="${FC_EXE:-gfortran}"
# Discover libraries (fallback to /usr)
export NETCDF="${NETCDF:-$(nc-config --prefix 2>/dev/null || echo /usr)}"
export NETCDF_FORTRAN="${NETCDF_FORTRAN:-$(nf-config --prefix 2>/dev/null || echo /usr)}"
export HDF5_ROOT="${HDF5_ROOT:-$(h5cc -showconfig 2>/dev/null | awk -F': ' "/Installation point/{print $2}" || echo /usr)}"
# Threads
export NCORES="${NCORES:-4}"
    '''.strip()


def get_external_tools_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Define all external tools required by CONFLUENCE.
    
    Returns:
        Dictionary mapping tool names to their complete configuration including:
        - description: Human-readable description
        - config_path_key: Key in config file for installation path
        - config_exe_key: Key in config file for executable name
        - default_path_suffix: Default relative path for installation
        - default_exe: Default executable/library filename
        - repository: Git repository URL (None for non-git installs)
        - branch: Git branch to checkout (None for default)
        - install_dir: Directory name for installation
        - requires: List of tool dependencies (other tools)
        - build_commands: Shell commands for building
        - dependencies: System dependencies required
        - test_command: Command argument for testing (None to skip)
        - verify_install: Installation verification criteria
        - order: Installation order (lower numbers first)
    """
    common_env = get_common_build_environment()
    
    return {
        # ================================================================
        # SUNDIALS - Solver Library (Install First - Required by SUMMA)
        # ================================================================
        'sundials': {
            'description': 'SUNDIALS - SUite of Nonlinear and DIfferential/ALgebraic equation Solvers',
            'config_path_key': 'SUNDIALS_INSTALL_PATH',
            'config_exe_key': 'SUNDIALS_DIR',
            'default_path_suffix': 'installs/sundials/install/sundials',
            'default_exe': 'lib64/libsundials_core.a',
            'repository': None,
            'branch': None,
            'install_dir': 'sundials',
            'build_commands': [
                r'''
# Build SUNDIALS from release tarball (shared libs OK; SUMMA will link).
SUNDIALS_VER=7.4.0
SUNDIALSDIR="$(pwd)/install/sundials"
rm -f "v${SUNDIALS_VER}.tar.gz" || true
wget -q https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz
tar -xzf v${SUNDIALS_VER}.tar.gz
cd sundials-${SUNDIALS_VER}
rm -rf build && mkdir build && cd build
cmake .. \
-DBUILD_FORTRAN_MODULE_INTERFACE=ON \
-DCMAKE_Fortran_COMPILER=gfortran \
-DCMAKE_INSTALL_PREFIX="$SUNDIALSDIR" \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_SHARED_LIBS=ON \
-DEXAMPLES_ENABLE=OFF \
-DBUILD_TESTING=OFF
cmake --build . --target install -j ${NCORES:-4}
# print lib dir for debug
[ -d "$SUNDIALSDIR/lib64" ] && ls -la "$SUNDIALSDIR/lib64" | head -20 || true
[ -d "$SUNDIALSDIR/lib" ] && ls -la "$SUNDIALSDIR/lib" | head -20 || true
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': [
                    'lib64/libsundials_core.a',
                    'lib/libsundials_core.a',
                    'include/sundials/sundials_config.h'
                ],
                'check_type': 'exists_any'
            },
            'order': 1
        },

        # ================================================================
        # SUMMA - Hydrological Model
        # ================================================================
        'summa': {
            'description': 'Structure for Unifying Multiple Modeling Alternatives (with SUNDIALS)',
            'config_path_key': 'SUMMA_INSTALL_PATH',
            'config_exe_key': 'SUMMA_EXE',
            'default_path_suffix': 'installs/summa/bin',
            'default_exe': 'summa.exe',
            'repository': 'https://github.com/CH-Earth/summa.git',
            'branch': 'develop_sundials',
            'install_dir': 'summa',
            'requires': ['sundials'],
            'build_commands': [
                r'''
# Build SUMMA against SUNDIALS + NetCDF, leverage SUMMA's build scripts pattern
set -e
export SUNDIALS_DIR="$(realpath ../sundials/install/sundials)"
echo "Using SUNDIALS from: $SUNDIALS_DIR"

# Determine LAPACK strategy based on platform
SPECIFY_LINKS=OFF

# macOS: Use manual LAPACK specification (Homebrew OpenBLAS isn't reliably detected by CMake)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected - using manual LAPACK specification"
    SPECIFY_LINKS=ON
    export LIBRARY_LINKS='-llapack'
# HPC with OpenBLAS module loaded
elif command -v module >/dev/null 2>&1 && module list 2>&1 | grep -qi openblas; then
    echo "OpenBLAS module loaded - using auto-detection"
    SPECIFY_LINKS=OFF
# Linux with system OpenBLAS
elif pkg-config --exists openblas 2>/dev/null || [ -f "/usr/lib64/libopenblas.so" ] || [ -f "/usr/lib/libopenblas.so" ]; then
    echo "System OpenBLAS found - using auto-detection"
    SPECIFY_LINKS=OFF
else
    # Fallback to manual LAPACK
    echo "Using manual LAPACK specification"
    SPECIFY_LINKS=ON
    export LIBRARY_LINKS='-llapack -lblas'
fi

rm -rf cmake_build && mkdir -p cmake_build
cmake -S build -B cmake_build \
-DUSE_SUNDIALS=ON \
-DCMAKE_BUILD_TYPE=Release \
-DSPECIFY_LAPACK_LINKS=$SPECIFY_LINKS \
-DCMAKE_PREFIX_PATH="$SUNDIALS_DIR" \
-DSUNDIALS_ROOT="$SUNDIALS_DIR" \
-DNETCDF_PATH="${NETCDF:-/usr}" \
-DNETCDF_FORTRAN_PATH="${NETCDF_FORTRAN:-/usr}" \
-DCMAKE_Fortran_COMPILER=gfortran \
-DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none"

# Build all targets (repo scripts use 'all', not 'summa_sundials')
cmake --build cmake_build --target all -j ${NCORES:-4}

# Create symlink - binary goes directly to bin/, not cmake_build/bin/
if [ -f "bin/summa_sundials.exe" ]; then
    cd bin && ln -sf summa_sundials.exe summa.exe && cd ..
elif [ -f "cmake_build/bin/summa_sundials.exe" ]; then
    mkdir -p bin
    cp cmake_build/bin/summa_sundials.exe bin/
    cd bin && ln -sf summa_sundials.exe summa.exe && cd ..
elif [ -f "cmake_build/bin/summa.exe" ]; then
    mkdir -p bin
    cp cmake_build/bin/summa.exe bin/
fi
                '''
            ],
            'dependencies': [],
            'test_command': '--version',
            'verify_install': {
                'file_paths': ['bin/summa.exe', 'bin/summa_sundials.exe', 'cmake_build/bin/summa.exe', 'cmake_build/bin/summa_sundials.exe'],
                'check_type': 'exists_any'
            },
            'order': 2
        },

        # ================================================================
        # mizuRoute - River Network Routing
        # ================================================================
        'mizuroute': {
            'description': 'Mizukami routing model for river network routing',
            'config_path_key': 'INSTALL_PATH_MIZUROUTE',
            'config_exe_key': 'EXE_NAME_MIZUROUTE',
            'default_path_suffix': 'installs/mizuRoute/route/bin',
            'default_exe': 'mizuRoute.exe',
            'repository': 'https://github.com/ESCOMP/mizuRoute.git',
            'branch': 'serial',
            'install_dir': 'mizuRoute',
            'build_commands': [
                common_env,
                r'''
# Build mizuRoute - edit Makefile directly (it doesn't use env vars)
cd route/build

# Create bin directory if it doesn't exist
mkdir -p ../bin

# F_MASTER should point to the route directory (one level up from build)
F_MASTER_PATH="$(cd .. && pwd)"
echo "F_MASTER will be set to: $F_MASTER_PATH/"

# Detect NetCDF Fortran - this is critical for compiling
echo "=== NetCDF Detection ==="
echo "NETCDF from common_env: ${NETCDF}"
echo "NETCDF_FORTRAN from common_env: ${NETCDF_FORTRAN}"

# Try nf-config first (NetCDF Fortran config tool)
if command -v nf-config >/dev/null 2>&1; then
    NETCDF_TO_USE="$(nf-config --prefix)"
    echo "Found nf-config, using: ${NETCDF_TO_USE}"
elif [ -n "${NETCDF_FORTRAN}" ] && [ -d "${NETCDF_FORTRAN}/include" ]; then
    NETCDF_TO_USE="${NETCDF_FORTRAN}"
    echo "Using NETCDF_FORTRAN: ${NETCDF_TO_USE}"
elif [ -n "${NETCDF}" ] && [ -d "${NETCDF}/include" ]; then
    NETCDF_TO_USE="${NETCDF}"
    echo "Using NETCDF: ${NETCDF_TO_USE}"
else
    echo "WARNING: NetCDF not found via environment variables"
    # Try common Mac/Linux locations
    for try_path in /opt/homebrew/opt/netcdf-fortran /opt/homebrew/opt/netcdf /usr/local/opt/netcdf-fortran /usr/local/opt/netcdf /usr/local /usr; do
        if [ -d "$try_path/include" ]; then
            NETCDF_TO_USE="$try_path"
            echo "Found NetCDF at: $NETCDF_TO_USE"
            break
        fi
    done
fi

# Final check
if [ -z "${NETCDF_TO_USE}" ]; then
    echo "ERROR: Could not find NetCDF installation"
    exit 1
fi

echo "Final NETCDF_TO_USE: ${NETCDF_TO_USE}"
echo "Checking include directory:"
ls -la "${NETCDF_TO_USE}/include/" 2>/dev/null | grep -E "netcdf|NETCDF" | head -5 || echo "Could not list includes"

# Also find the NetCDF C library (different from Fortran on Mac)
echo "=== Finding NetCDF C library ==="
if command -v nc-config >/dev/null 2>&1; then
    NETCDF_C_PATH="$(nc-config --prefix)"
    echo "Found nc-config, NetCDF C at: ${NETCDF_C_PATH}"
elif [ -d "/opt/homebrew/opt/netcdf" ]; then
    NETCDF_C_PATH="/opt/homebrew/opt/netcdf"
    echo "Found NetCDF C at: ${NETCDF_C_PATH}"
else
    NETCDF_C_PATH="${NETCDF_TO_USE}"
    echo "Using same path for NetCDF C: ${NETCDF_C_PATH}"
fi

# Show original Makefile state - cast wide net
echo "=== Makefile configuration (BEFORE editing) ==="
grep -E "^(FC|FC_EXE|EXE|F_MASTER|NCDF|isOpenMP)" Makefile | head -20
echo "=== Makefile lines 30-50 (for debugging) ==="
sed -n '30,50p' Makefile

# Edit the Makefile in-place - try multiple possible variable names
echo "=== Editing Makefile ==="
perl -i -pe "s|^FC\s*=\s*$|FC = gnu|" Makefile
perl -i -pe "s|^FC_EXE\s*=\s*$|FC_EXE = ${FC_EXE:-gfortran}|" Makefile
perl -i -pe "s|^EXE\s*=\s*$|EXE = mizuRoute.exe|" Makefile
perl -i -pe "s|^F_MASTER\s*=.*$|F_MASTER = $F_MASTER_PATH/|" Makefile
# CRITICAL: NCDF_PATH has leading whitespace in ifeq blocks - match any leading space
perl -i -pe "s|^\s*NCDF_PATH\s*=.*$| NCDF_PATH = ${NETCDF_TO_USE}|" Makefile
perl -i -pe "s|^isOpenMP\s*=.*$|isOpenMP = no|" Makefile

# Fix LIBNETCDF to include both netcdf-fortran and netcdf C library
# This handles Mac where they're separate Homebrew installs
if [ "${NETCDF_C_PATH}" != "${NETCDF_TO_USE}" ]; then
    echo "Fixing LIBNETCDF to include both netcdf-fortran and netcdf C paths"
    # Multi-line replacement to handle continuation
    perl -i -0777 -pe "s|LIBNETCDF = -Wl,-rpath,\\\$\(NCDF_PATH\)/lib[^\n]*\\\\\n[^\n]*|LIBNETCDF = -Wl,-rpath,${NETCDF_TO_USE}/lib -Wl,-rpath,${NETCDF_C_PATH}/lib -L${NETCDF_TO_USE}/lib -L${NETCDF_C_PATH}/lib -lnetcdff -lnetcdf|s" Makefile
fi

# Show what we set
echo "=== Makefile configuration (AFTER editing) ==="
grep -E "^(FC|FC_EXE|EXE|F_MASTER|NCDF|isOpenMP)" Makefile | head -20

# Clean any previous builds
make clean || true

# Build with explicit -j value
JOBS="${NCORES:-4}"
echo "Building with $JOBS parallel jobs..."
# Build - may fail on install step but exe is created
make -j "$JOBS" 2>&1 | tee build.log || true

# Check and move executable regardless of make exit code
if [ -f "mizuRoute.exe" ]; then
    echo "Build successful - moving executable to ../bin/"
    mv mizuRoute.exe ../bin/
    ls -la ../bin/
else
    echo "ERROR: No executable found in build directory"
    ls -la . || true
    exit 1
fi
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['route/bin/mizuRoute.exe'],
                'check_type': 'exists'
            },
            'order': 3
        },

        # ================================================================
        # FUSE - Framework for Understanding Structural Errors
        # ================================================================
        "fuse": {
            "description": "Framework for Understanding Structural Errors",
            "config_path_key": "FUSE_INSTALL_PATH",
            "config_exe_key": "FUSE_EXE",
            "default_path_suffix": "installs/fuse/bin",
            "default_exe": "fuse.exe",
            "repository": "https://github.com/CH-Earth/fuse.git",
            "branch": None,
            "install_dir": "fuse",
            "build_commands": [
                r'''
set -e

# --- Compiler configuration ---
export FC="${FC:-gfortran}"
export FC_EXE="${FC_EXE:-gfortran}"

# --- NetCDF detection with multiple fallback strategies ---
# Try nf-config first, then nc-config, then environment variables, then Homebrew, then common paths
if command -v nf-config >/dev/null 2>&1; then
  export NETCDF_FORTRAN="$(nf-config --prefix 2>/dev/null)"
elif command -v nc-config >/dev/null 2>&1; then
  export NETCDF_FORTRAN="$(nc-config --prefix 2>/dev/null)"
elif [ -n "$NETCDF_FORTRAN" ]; then
  : # Use existing environment variable
elif command -v brew >/dev/null 2>&1 && brew --prefix netcdf-fortran >/dev/null 2>&1; then
  export NETCDF_FORTRAN="$(brew --prefix netcdf-fortran)"
else
  # Common HPC module paths
  for path in /usr $HOME/.local /opt/netcdf-fortran /opt/netcdf; do
    if [ -d "$path/include" ] && [ -d "$path/lib" ]; then
      export NETCDF_FORTRAN="$path"
      break
    fi
  done
fi
export NETCDF_FORTRAN="${NETCDF_FORTRAN:-/usr}"

# NetCDF C library (often same as Fortran, but not always)
if command -v nc-config >/dev/null 2>&1; then
  export NETCDF="$(nc-config --prefix 2>/dev/null)"
elif [ -n "$NETCDF" ]; then
  : # Use existing environment variable
elif command -v brew >/dev/null 2>&1 && brew --prefix netcdf >/dev/null 2>&1; then
  export NETCDF="$(brew --prefix netcdf)"
else
  export NETCDF="$NETCDF_FORTRAN"
fi

# --- HDF5 detection with robust fallbacks ---
if command -v h5cc >/dev/null 2>&1; then
  export HDF5_ROOT="$(h5cc -showconfig 2>/dev/null | grep -i "Installation point" | sed 's/.*: *//' | head -n1)"
fi
if [ -z "$HDF5_ROOT" ] || [ ! -d "$HDF5_ROOT" ]; then
  if [ -n "$HDF5_ROOT" ]; then
    : # Use existing environment variable
  elif command -v brew >/dev/null 2>&1 && brew --prefix hdf5 >/dev/null 2>&1; then
    export HDF5_ROOT="$(brew --prefix hdf5)"
  else
    # Try common paths
    for path in /usr $HOME/.local /opt/hdf5; do
      if [ -d "$path/include" ] && [ -d "$path/lib" ]; then
        export HDF5_ROOT="$path"
        break
      fi
    done
  fi
fi
export HDF5_ROOT="${HDF5_ROOT:-/usr}"

# Map to FUSE Makefile variable names
export NCDF_PATH="$NETCDF_FORTRAN"
export HDF_PATH="$HDF5_ROOT"

# --- Platform-specific linker flags ---
if command -v brew >/dev/null 2>&1; then
  # macOS with Homebrew
  export CPPFLAGS="${CPPFLAGS:+$CPPFLAGS }-I$(brew --prefix netcdf)/include -I$(brew --prefix netcdf-fortran)/include"
  export LDFLAGS="${LDFLAGS:+$LDFLAGS }-L$(brew --prefix netcdf)/lib -L$(brew --prefix netcdf-fortran)/lib"
  if [ -d "$HDF_PATH/include" ]; then
    export CPPFLAGS="${CPPFLAGS} -I${HDF_PATH}/include"
    export LDFLAGS="${LDFLAGS} -L${HDF_PATH}/lib"
  fi
fi

# --- Warning for toolchain mismatches (non-fatal) ---
FC_VER=$("${FC}" -dumpfullversion 2>/dev/null || "${FC}" --version 2>/dev/null | head -n1 || echo "unknown")
echo "ℹ️  Compiler: ${FC} (${FC_VER})"
echo "ℹ️  NetCDF-Fortran: ${NETCDF_FORTRAN}"
# Check for major version mismatch but only warn
case "$NETCDF_FORTRAN" in 
  *gcc-[0-9]*)
    NETCDF_GCC_VER=$(echo "$NETCDF_FORTRAN" | grep -o 'gcc-[0-9][0-9]*' | cut -d- -f2)
    FC_MAJOR=$(echo "$FC_VER" | grep -o '^[0-9][0-9]*' || echo "0")
    if [ -n "$NETCDF_GCC_VER" ] && [ -n "$FC_MAJOR" ] && [ "$NETCDF_GCC_VER" != "$FC_MAJOR" ]; then
      echo "⚠️  Warning: NetCDF built with GCC ${NETCDF_GCC_VER} but using compiler version ${FC_MAJOR}"
      echo "   This may cause issues. Consider loading matching modules if available."
    fi
  ;;
esac

# --- Build ---
cd build
make clean 2>/dev/null || true
export F_MASTER="$(cd .. && pwd)/"

echo ""
echo "🔨 Build environment:"
echo "   FC: ${FC}"
echo "   F_MASTER: ${F_MASTER}"
echo "   NCDF_PATH: ${NCDF_PATH}"
echo "   HDF_PATH: ${HDF_PATH}"
echo ""

# Construct library and include paths to override Makefile's hardcoded values
# Find the actual lib directory (could be lib or lib64)
if [ -d "${NCDF_PATH}/lib64" ]; then
  NCDF_LIB_DIR="${NCDF_PATH}/lib64"
elif [ -d "${NCDF_PATH}/lib" ]; then
  NCDF_LIB_DIR="${NCDF_PATH}/lib"
else
  NCDF_LIB_DIR="${NCDF_PATH}/lib"
fi

if [ -d "${HDF_PATH}/lib64" ]; then
  HDF_LIB_DIR="${HDF_PATH}/lib64"
elif [ -d "${HDF_PATH}/lib" ]; then
  HDF_LIB_DIR="${HDF_PATH}/lib"
else
  HDF_LIB_DIR="${HDF_PATH}/lib"
fi

# On Mac/Homebrew, NetCDF C and NetCDF-Fortran are separate packages
# Add NetCDF C library path if it's different from NetCDF-Fortran
NETCDF_C_LIB=""
if command -v brew >/dev/null 2>&1; then
  NETCDF_C_PATH="$(brew --prefix netcdf 2>/dev/null || echo "")"
  if [ -n "$NETCDF_C_PATH" ] && [ "$NETCDF_C_PATH" != "$NCDF_PATH" ]; then
    NETCDF_C_LIB="-L${NETCDF_C_PATH}/lib"
  fi
fi

# Override the Makefile's LIBRARIES and INCLUDE variables
LIBRARIES="-L${NCDF_LIB_DIR} ${NETCDF_C_LIB} -lnetcdff -lnetcdf -L${HDF_LIB_DIR} -lhdf5_hl -lhdf5"
INCLUDE="-I${NCDF_PATH}/include -I${HDF_PATH}/include"

# Add legacy compiler flags for compatibility with old Fortran code
# -fallow-argument-mismatch: allows rank/type mismatches (needed for NetCDF calls)
# -std=legacy: allows old Fortran features like PAUSE
EXTRA_FLAGS="-fallow-argument-mismatch -std=legacy"
FLAGS_NORMA="-O3 -ffree-line-length-none -fmax-errors=0 -cpp ${EXTRA_FLAGS}"
FLAGS_FIXED="-O2 -c -ffixed-form ${EXTRA_FLAGS}"

echo "🔧 Library flags:"
echo "   LIBRARIES: ${LIBRARIES}"
echo "   INCLUDE: ${INCLUDE}"
echo "   FLAGS_NORMA: ${FLAGS_NORMA}"
echo ""

# Build sce_16plus.o first separately to avoid the broken Makefile rule
# We're already in the build directory from the earlier 'cd build' command
echo "🔨 Pre-compiling sce_16plus.f..."
${FC} ${FLAGS_FIXED} -o sce_16plus.o "FUSE_SRC/FUSE_SCE/sce_16plus.f" || {
  echo "❌ Failed to compile sce_16plus.f"
  exit 1
}
echo "✅ sce_16plus.o compiled"

# Now run make with overrides (without -j to avoid race conditions)
echo "🔨 Building FUSE..."
if make \
  FC="${FC}" \
  F_MASTER="${F_MASTER}" \
  LIBRARIES="${LIBRARIES}" \
  INCLUDE="${INCLUDE}" \
  FLAGS_NORMA="${FLAGS_NORMA}" \
  FLAGS_FIXED="${FLAGS_FIXED}"; then
  echo "✅ Build completed"
else
  echo "❌ Build failed"
  echo ""
  echo "Diagnostics:"
  echo "   NetCDF lib: ${NCDF_LIB_DIR}"
  echo "   HDF5 lib: ${HDF_LIB_DIR}"
  echo "   NetCDF includes: $(ls ${NCDF_PATH}/include/netcdf*.mod 2>/dev/null | head -3 || echo 'not found')"
  echo "   NetCDF libs: $(ls ${NCDF_LIB_DIR}/libnetcdff.* 2>/dev/null | head -3 || echo 'not found')"
  exit 1
fi

# Check if binary was created (we're in build dir, so bin is ../bin)
if [ -f ../bin/fuse.exe ]; then
  echo "✅ Binary in ../bin/fuse.exe"
elif [ -f fuse.exe ]; then
  echo "✅ Binary built in build dir, staging to ../bin/"
  mkdir -p ../bin
  cp fuse.exe ../bin/
  echo "✅ Binary staged to ../bin/fuse.exe"
else
  echo "❌ fuse.exe not found after build"
  echo "Checking build directory contents:"
  ls -la . 2>/dev/null | grep -E "\.(exe|out|o)$" || echo "No executables found"
  exit 1
fi

# Verify the binary (we're in build dir)
if [ -f ../bin/fuse.exe ]; then
  echo ""
  echo "🧪 Testing binary..."
  ../bin/fuse.exe 2>&1 | head -5 || true
  echo "✅ FUSE build successful"
else
  echo "❌ Verification failed: ../bin/fuse.exe not found"
  exit 1
fi
        '''
        ],
        "dependencies": [],
        "test_command": None,
        "verify_install": {
            "file_paths": ["bin/fuse.exe"],
            "check_type": "exists"
        },
        "order": 4
            },


        # ================================================================
        # TauDEM - Terrain Analysis
        # ================================================================
        'taudem': {
            'description': 'Terrain Analysis Using Digital Elevation Models',
            'config_path_key': 'TAUDEM_INSTALL_PATH',
            'config_exe_key': 'TAUDEM_EXE',
            'default_path_suffix': 'installs/TauDEM/bin',
            'default_exe': 'pitremove',
            'repository': 'https://github.com/dtarb/TauDEM.git',
            'branch': None,
            'install_dir': 'TauDEM',
            'build_commands': [
                r'''
set -e
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -S .. -B .
cmake --build . -j ${NCORES:-4}
mkdir -p ../bin
# copy common executables
find . -maxdepth 1 -type f -perm -111 -exec cp {} ../bin/ \; || true
                '''
            ],
            'dependencies': [],
            'test_command': '--help',
            'verify_install': {
                'file_paths': ['bin/pitremove'],
                'check_type': 'exists'
            },
            'order': 5
        },

        # ================================================================
        # GIStool - Geospatial Data Extraction
        # ================================================================
        'gistool': {
            'description': 'Geospatial data extraction and processing tool',
            'config_path_key': 'INSTALL_PATH_GISTOOL',
            'config_exe_key': 'EXE_NAME_GISTOOL',
            'default_path_suffix': 'installs/gistool',
            'default_exe': 'extract-gis.sh',
            'repository': 'https://github.com/kasra-keshavarz/gistool.git',
            'branch': None,
            'install_dir': 'gistool',
            'build_commands': [
                r'''
set -e
chmod +x extract-gis.sh
                '''
            ],
            'verify_install': {
                'file_paths': ['extract-gis.sh'],
                'check_type': 'exists'
            },
            'dependencies': [],
            'test_command': None,
            'order': 6
        },

        # ================================================================
        # Datatool - Meteorological Data Processing
        # ================================================================
        'datatool': {
            'description': 'Meteorological data extraction and processing tool',
            'config_path_key': 'DATATOOL_PATH',
            'config_exe_key': 'DATATOOL_SCRIPT',
            'default_path_suffix': 'installs/datatool',
            'default_exe': 'extract-dataset.sh',
            'repository': 'https://github.com/kasra-keshavarz/datatool.git',
            'branch': None,
            'install_dir': 'datatool',
            'build_commands': [
                r'''
set -e
chmod +x extract-dataset.sh
                '''
            ],
            'dependencies': [],
            'test_command': '--help',
            'verify_install': {
                'file_paths': ['extract-dataset.sh'],
                'check_type': 'exists'
            },
            'order': 7
        },

        # ================================================================
        # NGEN - NextGen National Water Model Framework
        # ================================================================
        'ngen': {
            'description': 'NextGen National Water Model Framework',
            'config_path_key': 'NGEN_INSTALL_PATH',
            'config_exe_key': 'NGEN_EXE',
            'default_path_suffix': 'installs/ngen/cmake_build',
            'default_exe': 'ngen',
            'repository': 'https://github.com/CIROH-UA/ngen',
            'branch': 'ngiab',
            'install_dir': 'ngen',
            'build_commands': [
                r'''
set -e
echo "Building ngen..."

# Make sure CMake sees a supported NumPy, and ignore user-site
export PYTHONNOUSERSITE=1
python -m pip install --upgrade "pip<24.1" >/dev/null 2>&1 || true
python - <<'PY' || (python -m pip install "numpy<2" "setuptools<70" && true)
from packaging.version import Version
import numpy as np
assert Version(np.__version__) < Version("2.0")
PY
python - <<'PY'
import numpy as np
print("Using NumPy:", np.__version__)
PY

# Boost (local)
if [ ! -d "boost_1_79_0" ]; then
echo "Fetching Boost 1.79.0..."
(wget -q https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2 -O boost_1_79_0.tar.bz2 \
    || curl -fsSL -o boost_1_79_0.tar.bz2 https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2)
tar -xjf boost_1_79_0.tar.bz2 && rm -f boost_1_79_0.tar.bz2
fi
export BOOST_ROOT="$(pwd)/boost_1_79_0"
export CXX=${CXX:-g++}

# Submodules needed
git submodule update --init --recursive -- test/googletest extern/pybind11 || true

rm -rf cmake_build

# First try with Python ON
if cmake -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT="$BOOST_ROOT" -DNGEN_WITH_PYTHON=ON -S . -B cmake_build; then
echo "Configured with Python ON"
else
echo "CMake failed with Python ON; retrying with Python OFF"
rm -rf cmake_build
cmake -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT="$BOOST_ROOT" -DNGEN_WITH_PYTHON=OFF -S . -B cmake_build
fi

cmake --build cmake_build --target ngen -j ${NCORES:-4}
./cmake_build/ngen --help >/dev/null || true
                '''
            ],
            'dependencies': [],
            'test_command': '--help',
            'verify_install': {
                'file_paths': ['cmake_build/ngen'],
                'check_type': 'exists'
            },
            'order': 8
        },

        # ================================================================
        # NGIAB - NextGen In A Box
        # ================================================================
        'ngiab': {
            'description': 'NextGen In A Box - Container-based ngen deployment',
            'config_path_key': 'NGIAB_INSTALL_PATH',
            'config_exe_key': 'NGIAB_SCRIPT',
            'default_path_suffix': 'installs/ngiab',
            'default_exe': 'guide.sh',
            'repository': None,
            'branch': 'main',
            'install_dir': 'ngiab',
            'build_commands': [
                r'''
set -e
# Detect HPC vs laptop/workstation and fetch the right NGIAB wrapper repo into ../ngiab
IS_HPC=false
for scheduler in sbatch qsub bsub; do
if command -v $scheduler >/dev/null 2>&1; then IS_HPC=true; break; fi
done
[ -n "$SLURM_CLUSTER_NAME" ] && IS_HPC=true
[ -n "$PBS_JOBID" ] && IS_HPC=true
[ -n "$SGE_CLUSTER_NAME" ] && IS_HPC=true
[ -d "/scratch" ] && IS_HPC=true

if $IS_HPC; then
NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-HPCInfra.git"
echo "HPC environment detected; using NGIAB-HPCInfra"
else
NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-CloudInfra.git"
echo "Non-HPC environment detected; using NGIAB-CloudInfra"
fi

cd ..
rm -rf ngiab
git clone "$NGIAB_REPO" ngiab
cd ngiab
[ -f guide.sh ] && chmod +x guide.sh && bash -n guide.sh || true
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['guide.sh'],
                'check_type': 'exists'
            },
            'order': 9
        },
    }


if __name__ == "__main__":
    """Test the configuration definitions."""
    tools = get_external_tools_definitions()
    print(f"✅ Loaded {len(tools)} external tool definitions:")
    for name, info in sorted(tools.items(), key=lambda x: x[1]['order']):
        print(f"   {info['order']:2d}. {name:12s} - {info['description'][:60]}")