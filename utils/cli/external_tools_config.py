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
# Build SUMMA against SUNDIALS + NetCDF, prefer OpenBLAS on macOS/Homebrew
set -e
SUNDIALS_DIR="$(realpath ../sundials/install/sundials)"
echo "Using SUNDIALS from: $SUNDIALS_DIR"
if [ -d "$SUNDIALS_DIR/lib64" ]; then SUND_LIB="$SUNDIALS_DIR/lib64"; else SUND_LIB="$SUNDIALS_DIR/lib"; fi

# Homebrew OpenBLAS (mac)
OPENBLAS_ARG=""
if command -v brew >/dev/null 2>&1 && [ -d "$(brew --prefix)/opt/openblas" ]; then
OPENBLAS_ARG="-DBLA_VENDOR=OpenBLAS -DOpenBLAS_DIR=$(brew --prefix)/opt/openblas"
fi

rm -rf cmake_build && mkdir -p cmake_build
cmake -S build -B cmake_build \
-DUSE_SUNDIALS=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH="$SUNDIALS_DIR" \
-DSUNDIALS_ROOT="$SUNDIALS_DIR" \
-DNETCDF_PATH="${NETCDF:-/usr}" \
-DNETCDF_FORTRAN_PATH="${NETCDF_FORTRAN:-/usr}" \
$OPENBLAS_ARG \
-DCMAKE_Fortran_COMPILER=gfortran \
-DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none"

cmake --build cmake_build --target summa_sundials -j ${NCORES:-4}

mkdir -p bin
if [ -f "cmake_build/bin/summa_sundials.exe" ]; then
cp cmake_build/bin/summa_sundials.exe bin/
ln -sf summa_sundials.exe bin/summa.exe
elif [ -f "cmake_build/bin/summa.exe" ]; then
cp cmake_build/bin/summa.exe bin/
fi
                '''
            ],
            'dependencies': [],
            'test_command': '--version',
            'verify_install': {
                'file_paths': ['bin/summa.exe', 'cmake_build/bin/summa.exe', 'cmake_build/bin/summa_sundials.exe'],
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
# Build without editing Makefiles; satisfy FC sanity check
cd route/build
make clean || true
make -j "${NCORES}" \
FC="${FC}" FC_EXE="${FC_EXE}" \
NETCDF="${NETCDF}" NETCDF_FORTRAN="${NETCDF_FORTRAN}"

# show outputs
ls -la ../bin/ || true
# accept any of the known exe names
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['route/bin/mizuRoute.exe', 'route/bin/mizuroute.exe', 'route/bin/runoff_route.exe'],
                'check_type': 'exists_any'
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

# Run the FUSE build - try multiple approaches
# First try: just 'make' without 'all' target
if make FC="${FC}" F_MASTER="${F_MASTER}" NCDF_PATH="${NCDF_PATH}" HDF_PATH="${HDF_PATH}" -j "${NCORES:-4}"; then
  echo "✅ Build completed with 'make'"
elif make install FC="${FC}" F_MASTER="${F_MASTER}" NCDF_PATH="${NCDF_PATH}" HDF_PATH="${HDF_PATH}" -j "${NCORES:-4}"; then
  echo "✅ Build completed with 'make install'"  
else
  echo "❌ Make failed. Checking for common issues..."
  echo "   NetCDF includes: $(find ${NCDF_PATH}/include -name "netcdf*.mod" 2>/dev/null | head -3 || echo 'not found')"
  echo "   NetCDF libs: $(ls ${NCDF_PATH}/lib*/libnetcdff.* 2>/dev/null | head -3 || echo 'not found')"
  echo ""
  echo "   Checking what Make is trying to build:"
  make -n FC="${FC}" F_MASTER="${F_MASTER}" NCDF_PATH="${NCDF_PATH}" HDF_PATH="${HDF_PATH}" 2>&1 | head -10
  exit 1
fi

# Stage binary to ../bin/
mkdir -p ../bin
if [ -f fuse.exe ]; then
  cp fuse.exe ../bin/
  echo "✅ Binary staged to ../bin/fuse.exe"
elif [ -f ../bin/fuse.exe ]; then
  echo "✅ Binary already in ../bin/fuse.exe"
else
  echo "❌ fuse.exe not found after build"
  ls -la . 2>/dev/null || true
  exit 1
fi

# Verify the binary
if [ -f ../bin/fuse.exe ]; then
  echo ""
  echo "🧪 Testing binary..."
  ../bin/fuse.exe 2>&1 | head -5 || true
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