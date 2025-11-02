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
        'fuse': {
            'description': 'Framework for Understanding Structural Errors',
            'config_path_key': 'FUSE_INSTALL_PATH',
            'config_exe_key': 'FUSE_EXE',
            'default_path_suffix': 'installs/fuse/bin',
            'default_exe': 'fuse.exe',
            'repository': 'https://github.com/naddor/fuse.git',
            'branch': None,
            'install_dir': 'fuse',
            'build_commands': [
                common_env,
                r'''
# Build FUSE executable using the upstream Makefile
cd build
make clean || true

# Set F_MASTER to the fuse root (parent of build/)
export F_MASTER="$(cd .. && pwd)/"

# Find HDF5 on HPC systems (check common HPC locations)
if [ -z "$HDF5_ROOT" ]; then
    # Try module environment first
    for var in HDF5_ROOT HDF5_DIR HDF5HOME EBROOTHDF5; do
        if [ -n "${!var}" ]; then
            export HDF5_ROOT="${!var}"
            echo "Found HDF5 via \$${var}: $HDF5_ROOT"
            break
        fi
    done
    
    # If still not found, try to detect from loaded modules or common paths
    if [ -z "$HDF5_ROOT" ]; then
        for path in /apps/spack/*/apps/hdf5/*/lib /usr/lib64/hdf5 /usr/lib/x86_64-linux-gnu/hdf5/serial; do
            if [ -d "$path" ]; then
                export HDF5_ROOT="$(dirname "$path")"
                echo "Detected HDF5 at: $HDF5_ROOT"
                break
            fi
        done
    fi
    
    # Last resort fallback
    HDF5_ROOT="${HDF5_ROOT:-/usr}"
fi

echo "Build environment:"
echo "  FC: ${FC}"
echo "  F_MASTER: ${F_MASTER}"
echo "  NETCDF_FORTRAN: ${NETCDF_FORTRAN}"
echo "  HDF5_ROOT: ${HDF5_ROOT}"

# Run the full build (make all handles compilation order correctly)
make all \
FC="${FC}" \
F_MASTER="${F_MASTER}" \
NCDF_LIB_PATH="${NETCDF_FORTRAN}/lib" \
HDF_LIB_PATH="${HDF5_ROOT}/lib" \
INCLUDE_PATH="${NETCDF_FORTRAN}"

# Verify executable was created
ls -la ../bin/ || true
if [ -f ../bin/fuse.exe ]; then
    echo "✅ fuse.exe built successfully"
    ../bin/fuse.exe 2>&1 | head -5 || true
else
    echo "⚠️ fuse.exe not found"
    exit 1
fi
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['bin/fuse.exe'],
                'check_type': 'exists'
            },
            'order': 4
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