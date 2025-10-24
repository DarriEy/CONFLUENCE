#!/usr/bin/env python3
"""
CONFLUENCE CLI Argument Manager Utility

Save this file as: utils/cli/cli_argument_manager.py

This utility provides comprehensive command-line interface functionality for the CONFLUENCE
hydrological modeling platform. It handles argument parsing, validation, and workflow
step execution control.

Features:
- Individual workflow step execution
- Pour point coordinate setup  
- Flexible configuration management
- Debug and logging controls
- Workflow validation and status reporting

Usage:
    from utils.cli.cli_argument_manager import CLIArgumentManager
    
    cli_manager = CLIArgumentManager()
    args = cli_manager.parse_arguments()
    plan = cli_manager.get_execution_plan(args)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml
import subprocess
import os
import shutil


class CLIArgumentManager:
    """
    Manages command-line arguments and workflow execution options for CONFLUENCE.
    
    This class provides a comprehensive CLI interface that allows users to:
    - Run individual workflow steps
    - Set up pour point configurations
    - Control workflow execution behavior
    - Manage configuration and debugging options
    
    The argument manager integrates with the existing CONFLUENCE workflow orchestrator
    to provide granular control over workflow execution.
    """
    
    def __init__(self):
        """Initialize the CLI argument manager."""
        self.parser = None
        self.workflow_steps = self._define_workflow_steps()
        self.domain_definition_methods = ['lumped', 'point', 'subset', 'delineate']
        self.external_tools = self._define_external_tools()  
        self._setup_parser()
        
    def _define_workflow_steps(self) -> Dict[str, Dict[str, str]]:
        """
        Define available workflow steps that can be run individually.
        
        Returns:
            Dictionary mapping step names to their descriptions and manager methods
        """
        return {
            'setup_project': {
                'description': 'Initialize project directory structure and shapefiles',
                'manager': 'project',
                'method': 'setup_project',
                'function_name': 'setup_project'
            },
            'create_pour_point': {
                'description': 'Create pour point shapefile from coordinates',
                'manager': 'project', 
                'method': 'create_pour_point',
                'function_name': 'create_pour_point'
            },
            'acquire_attributes': {
                'description': 'Download and process geospatial attributes (soil, land class, etc.)',
                'manager': 'data',
                'method': 'acquire_attributes',
                'function_name': 'acquire_attributes'
            },
            'define_domain': {
                'description': 'Define hydrological domain boundaries and river basins',
                'manager': 'domain',
                'method': 'define_domain',
                'function_name': 'define_domain'
            },
            'discretize_domain': {
                'description': 'Discretize domain into HRUs or other modeling units',
                'manager': 'domain',
                'method': 'discretize_domain',
                'function_name': 'discretize_domain'
            },
            'process_observed_data': {
                'description': 'Process observational data (streamflow, etc.)',
                'manager': 'data',
                'method': 'process_observed_data',
                'function_name': 'process_observed_data'
            },
            'acquire_forcings': {
                'description': 'Acquire meteorological forcing data',
                'manager': 'data',
                'method': 'acquire_forcings',
                'function_name': 'acquire_forcings'
            },
            'model_agnostic_preprocessing': {
                'description': 'Run model-agnostic preprocessing of forcing and attribute data',
                'manager': 'data',
                'method': 'model_agnostic_preprocessing',
                'function_name': 'model_agnostic_preprocessing'
            },
            'model_specific_preprocessing': {
                'description': 'Setup model-specific input files and configuration',
                'manager': 'model',
                'method': 'model_specific_preprocessing',
                'function_name': 'model_specific_preprocessing'
            },
            'run_model': {
                'description': 'Execute the hydrological model simulation',
                'manager': 'model',
                'method': 'run_model',
                'function_name': 'run_model'
            },
            'calibrate_model': {
                'description': 'Run model calibration and parameter optimization',
                'manager': 'optimization',
                'method': 'run_calibration',
                'function_name': 'run_calibration'  # Note: actual function name differs from CLI name
            },
            'run_emulation': {
                'description': 'Run emulation-based optimization if configured',
                'manager': 'optimization',
                'method': 'run_emulation',
                'function_name': 'run_emulation'
            },
            'run_benchmarking': {
                'description': 'Run benchmarking analysis against observations',
                'manager': 'analysis',
                'method': 'run_benchmarking',
                'function_name': 'run_benchmarking'
            },
            'run_decision_analysis': {
                'description': 'Run decision analysis for model comparison',
                'manager': 'analysis',
                'method': 'run_decision_analysis',
                'function_name': 'run_decision_analysis'
            },
            'run_sensitivity_analysis': {
                'description': 'Run sensitivity analysis on model parameters',
                'manager': 'analysis',
                'method': 'run_sensitivity_analysis',
                'function_name': 'run_sensitivity_analysis'
            },
            'postprocess_results': {
                'description': 'Postprocess and finalize model results',
                'manager': 'model',
                'method': 'postprocess_results',
                'function_name': 'postprocess_results'
            }
        }


    def _define_external_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Define external tools/binaries required by CONFLUENCE.
        
        Returns:
            Dictionary with tool definitions including config keys, executables, and repositories
        """
        return {
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
'''
# Build SUNDIALS using release tarball with SHARED libraries
SUNDIALS_VER=7.4.0
SUNDIALSDIR="$(pwd)/install/sundials"

echo "Downloading SUNDIALS v${SUNDIALS_VER}..."
wget -q https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download SUNDIALS"
    exit 1
fi

echo "Extracting SUNDIALS..."
tar -xzf v${SUNDIALS_VER}.tar.gz

cd sundials-${SUNDIALS_VER}
mkdir -p build
cd build

echo "Configuring SUNDIALS with CMake (with shared libraries)..."
cmake .. \
    -DBUILD_FORTRAN_MODULE_INTERFACE=ON \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_INSTALL_PREFIX="$SUNDIALSDIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DEXAMPLES_ENABLE=OFF \
    -DBUILD_TESTING=OFF

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed"
    exit 1
fi

echo "Building SUNDIALS..."
make -j 4

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo "Installing SUNDIALS..."
make install

if [ $? -ne 0 ]; then
    echo "ERROR: Installation failed"
    exit 1
fi

echo "✅ SUNDIALS v${SUNDIALS_VER} installed to: $SUNDIALSDIR"

# Check which lib directory was created
if [ -d "$SUNDIALSDIR/lib64" ]; then
    echo "Libraries installed to lib64:"
    ls -la "$SUNDIALSDIR/lib64/" | head -10
elif [ -d "$SUNDIALSDIR/lib" ]; then
    echo "Libraries installed to lib:"
    ls -la "$SUNDIALSDIR/lib/" | head -10
else
    echo "WARNING: No lib or lib64 directory found"
fi
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
                'check_type': 'exists'
            },
            'order': 1
        },
        'summa': {
            'description': 'Structure for Unifying Multiple Modeling Alternatives (with SUNDIALS)',
            'config_path_key': 'SUMMA_INSTALL_PATH',
            'config_exe_key': 'SUMMA_EXE',
            'default_path_suffix': 'installs/summa/bin',
            'default_exe': 'summa.exe',
            'repository': 'https://github.com/NCAR/summa.git',
            'branch': 'develop_sundials',
            'install_dir': 'summa',
            'requires': ['sundials'],
            'build_commands': [
'''
# Build SUMMA with SUNDIALS
export SUNDIALS_DIR="$(realpath ../sundials/install/sundials)"

echo "Using SUNDIALS from: $SUNDIALS_DIR"

# Verify SUNDIALS installation exists
if [ -d "$SUNDIALS_DIR/lib64" ]; then
    echo "✅ Found SUNDIALS libraries in lib64/"
    SUNDIALS_LIB_DIR="$SUNDIALS_DIR/lib64"
elif [ -d "$SUNDIALS_DIR/lib" ]; then
    echo "✅ Found SUNDIALS libraries in lib/"
    SUNDIALS_LIB_DIR="$SUNDIALS_DIR/lib"
else
    echo "ERROR: SUNDIALS libraries not found at $SUNDIALS_DIR"
    exit 1
fi

# Check for Fortran module interface
if [ ! -f "$SUNDIALS_LIB_DIR/libsundials_fida_mod.a" ] && [ ! -f "$SUNDIALS_LIB_DIR/libsundials_fida_mod.so" ]; then
    echo "⚠️  Warning: SUNDIALS Fortran module interface may not be installed"
    echo "   Looking for libsundials_fida_mod in: $SUNDIALS_LIB_DIR"
    ls -la "$SUNDIALS_LIB_DIR/" | grep sundials | head -10
fi

# Create cmake_build directory
mkdir -p cmake_build
cd cmake_build

echo "Configuring SUMMA with CMake..."
echo "Source directory: ../build"
echo "Build directory: $(pwd)"

# Try multiple approaches to help CMake find SUNDIALS
export CMAKE_PREFIX_PATH="$SUNDIALS_DIR:$CMAKE_PREFIX_PATH"
export SUNDIALS_ROOT="$SUNDIALS_DIR"

# Set Fortran flags for free-form source format
export FFLAGS="-ffree-form -ffree-line-length-none"
export FCFLAGS="-ffree-form -ffree-line-length-none"

cmake ../build \
    -DUSE_SUNDIALS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$SUNDIALS_DIR" \
    -DSUNDIALS_ROOT="$SUNDIALS_DIR" \
    -DSUNDIALS_DIR="$SUNDIALS_DIR" \
    -DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none" \
    -DCMAKE_Fortran_COMPILER=gfortran

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: CMake configuration failed"
    echo ""
    echo "🔍 Debugging information:"
    echo "   SUNDIALS location: $SUNDIALS_DIR"
    echo "   Checking for CMake config files:"
    find "$SUNDIALS_DIR" -name "*.cmake" 2>/dev/null | head -10
    exit 1
fi

echo ""
echo "Fixing CMake build files to remove C++ flags from Fortran compilation..."
find . -name "*.make" -type f -exec sed -i 's/-cxxlib//g' {} \;
find . -name "flags.make" -type f -exec sed -i 's/-cxxlib//g' {} \;

echo ""
echo "Building SUMMA (this may take a few minutes)..."
cmake --build . --target all -j 4

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "Checking for SUMMA executable..."
cd ..

# Check for executable in various locations with both possible names
if [ -f "bin/summa_sundials.exe" ]; then
    echo "✅ SUMMA executable found at: $(pwd)/bin/summa_sundials.exe"
    ln -sf summa_sundials.exe bin/summa.exe
    echo "✅ Created symlink: bin/summa.exe -> bin/summa_sundials.exe"
elif [ -f "bin/summa.exe" ]; then
    echo "✅ SUMMA executable found at: $(pwd)/bin/summa.exe"
elif [ -f "cmake_build/bin/summa_sundials.exe" ]; then
    mkdir -p bin
    cp cmake_build/bin/summa_sundials.exe bin/
    ln -sf summa_sundials.exe bin/summa.exe
    echo "✅ SUMMA executable installed to: $(pwd)/bin/summa_sundials.exe"
    echo "✅ Created symlink: bin/summa.exe -> bin/summa_sundials.exe"
elif [ -f "cmake_build/bin/summa.exe" ]; then
    mkdir -p bin
    cp cmake_build/bin/summa.exe bin/
    echo "✅ SUMMA executable installed to: $(pwd)/bin/summa.exe"
else
    echo "ERROR: SUMMA executable not found after build"
    echo "Searching for summa files..."
    find . -name "summa*.exe" -o -name "summa" -type f 2>/dev/null || echo "No summa executables found"
    exit 1
fi
'''
],
            'dependencies': [],
            'test_command': '--version',
            'order': 2
        },
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
'''
# Get NetCDF paths
if [ -z "$NETCDF" ]; then
    NCDF_PATH=$(nc-config --prefix 2>/dev/null || echo "/usr")
    export NETCDF=$NCDF_PATH
    echo "Set NETCDF to: $NETCDF"
fi

if [ -z "$NETCDF_FORTRAN" ]; then
    NCDFF_PATH=$(nf-config --prefix 2>/dev/null || echo "/usr")
    export NETCDF_FORTRAN=$NCDFF_PATH
    echo "Set NETCDF_FORTRAN to: $NCDFF_PATH"
fi

# Navigate to route directory and build
cd route
make FC=gfortran -j 4

if [ $? -ne 0 ]; then
    echo "ERROR: mizuRoute build failed"
    exit 1
fi

echo "✅ mizuRoute built successfully"
ls -la bin/
'''
],
            'dependencies': [],
            'test_command': '--version',
            'order': 3
        },
        'fuse': {
            'description': 'Framework for Understanding Structural Errors',
            'config_path_key': 'INSTALL_PATH_FUSE',
            'config_exe_key': 'EXE_NAME_FUSE',
            'default_path_suffix': 'installs/fuse/bin',
            'default_exe': 'fuse.exe',
            'repository': 'https://github.com/naddor/fuse.git',
            'branch': None,
            'install_dir': 'fuse',
            'build_commands': [
'''
# Setup build environment for FUSE
FUSE_INSTALL_DIR=$(pwd)

# Get NetCDF paths
NCDF_PATH=$(nc-config --prefix 2>/dev/null || echo "/usr")
NCDFF_PATH=$(nf-config --prefix 2>/dev/null || echo "/usr")
HDF5_PATH=$(h5cc -showconfig 2>/dev/null | grep "Installation point:" | awk '{print $3}' || echo "/usr")

echo "NetCDF C path: $NCDF_PATH"
echo "NetCDF Fortran path: $NCDFF_PATH"
echo "HDF5 path: $HDF5_PATH"

# Navigate to build_unix directory
cd build/build_unix

# Create custom Makefile
cat > Makefile << 'MAKEFILE_END'
# FUSE Makefile - Auto-generated by CONFLUENCE

FC = gfortran
INCLUDES = -I NCDFF_PATH_PLACEHOLDER/include
LIBRARIES = -L NCDFF_PATH_PLACEHOLDER/lib -lnetcdff -L NCDF_PATH_PLACEHOLDER/lib -lnetcdf -L HDF5_PATH_PLACEHOLDER/lib -lhdf5_hl -lhdf5

FUSE_PATH = FUSE_INSTALL_DIR_PLACEHOLDER
DRIVER_DIR = $(FUSE_PATH)build/FUSE_SRC/FUSE_DMSL/
EXE_PATH = $(FUSE_PATH)bin/

# ... [rest of Makefile content - include the full makefile from original]
MAKEFILE_END

# Replace placeholders
sed -i "s|FUSE_INSTALL_DIR_PLACEHOLDER|$FUSE_INSTALL_DIR/|g" Makefile
sed -i "s|NCDFF_PATH_PLACEHOLDER|$NCDFF_PATH|g" Makefile
sed -i "s|NCDF_PATH_PLACEHOLDER|$NCDF_PATH|g" Makefile
sed -i "s|HDF5_PATH_PLACEHOLDER|$HDF5_PATH|g" Makefile

echo "Building FUSE..."
make all

if [ $? -ne 0 ]; then
    echo "ERROR: FUSE build failed"
    exit 1
fi

if [ -f "../bin/fuse.exe" ]; then
    echo "✅ FUSE executable successfully created"
else
    echo "ERROR: fuse.exe not found after build"
    exit 1
fi
'''
],
            'dependencies': [],
            'test_command': '--version',
            'order': 4
        },
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
'''
# Build TauDEM
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4

# Create bin directory and copy executables
mkdir -p ../bin
cp src/* ../bin/ 2>/dev/null || true

echo "✅ TauDEM executables in: $(pwd)/src/"
ls -la src/ | head -10
'''
],
            'dependencies': [],
            'test_command': '--help',
            'order': 5
        },
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
'''
echo "✅ gistool cloned successfully - no compilation needed"
if [ -f extract-gis.sh ]; then
    chmod +x extract-gis.sh
    echo "✅ extract-gis.sh found and made executable"
else
    echo "ERROR: extract-gis.sh not found"
    exit 1
fi
'''
],
            'verify_install': {
                'file_paths': ['extract-gis.sh'],
                'check_type': 'exists'
            },
            'dependencies': [],
            'order': 6
        },
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
'''
# Make datatool script executable
chmod +x extract-dataset.sh
echo "✅ datatool script is now executable"
'''
],
            'dependencies': [],
            'test_command': '--help',
            'order': 7
        },

        'ngen': {
    'description': 'NextGen National Water Model Framework',
    'config_path_key': 'NGEN_INSTALL_PATH',
    'config_exe_key': 'NGEN_EXE',
    'default_path_suffix': 'installs/ngen',
    'default_exe': 'build/ngen',
    'repository': 'https://github.com/CIROH-UA/ngen', #'https://github.com/NOAA-OWP/ngen.git', 
    'branch': 'master',
    'install_dir': 'ngen',
    'build_commands': [
'''
# Build ngen from source
echo "Building ngen..."

# Download Boost
if [ ! -d "boost_1_79_0" ]; then
    curl -L -o boost_1_79_0.tar.bz2 \
        https://sourceforge.net/projects/boost/files/boost/1.79.0/boost_1_79_0.tar.bz2/download
    tar -xjf boost_1_79_0.tar.bz2
    rm boost_1_79_0.tar.bz2
fi

# Set environment
export BOOST_ROOT="$(pwd)/boost_1_79_0"
export CXX=${CXX:-g++}

# Get submodules
git submodule update --init --recursive -- test/googletest 
git submodule update --init --recursive -- extern/pybind11

# Configure
cmake -DCMAKE_CXX_COMPILER=$CXX \
      -DBOOST_ROOT=$BOOST_ROOT \
      -B build \
      -S .

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed"
    exit 1
fi

# Build
NCORES=$(nproc 2>/dev/null || echo 4)
cmake --build build --target ngen -- -j $NCORES

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

# Verify
if [ -f "build/ngen" ]; then
    chmod +x build/ngen
    echo "✅ ngen built successfully"
else
    echo "ERROR: ngen executable not found"
    exit 1
fi
'''
    ],
    'dependencies': [],
    'test_command': '--help',
    'verify_install': {
        'file_paths': ['build/ngen'],
        'check_type': 'executable'
    },
    'order': 8
                },
        'ngiab': {
            'description': 'NextGen In A Box - Container-based ngen deployment',
            'config_path_key': 'NGIAB_INSTALL_PATH',
            'config_exe_key': 'NGIAB_SCRIPT',
            'default_path_suffix': 'installs/ngiab',
            'default_exe': 'guide.sh',
            'repository': None,  # Will be determined based on environment
            'branch': 'main',
            'install_dir': 'ngiab',
            'build_commands': [
'''
# Install NGIAB based on environment (HPC vs laptop)
echo "🔍 Detecting environment..."

# Detect environment
IS_HPC=false

# Check for HPC schedulers
for scheduler in sbatch qsub bsub; do
    if command -v $scheduler &> /dev/null; then
        IS_HPC=true
        echo "✅ Detected HPC scheduler: $scheduler"
        break
    fi
done

# Check for HPC environment variables
if [ -n "$SLURM_CLUSTER_NAME" ] || [ -n "$PBS_JOBID" ] || [ -n "$SGE_CLUSTER_NAME" ]; then
    IS_HPC=true
    echo "✅ Detected HPC environment variables"
fi

# Check for /scratch directory
if [ -d "/scratch" ] && [ "$IS_HPC" = "false" ]; then
    IS_HPC=true
    echo "✅ Detected /scratch directory (common on HPC)"
fi

# Determine which repository to use
if [ "$IS_HPC" = "true" ]; then
    NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-HPCInfra.git"
    echo "🖥️  HPC environment detected - using NGIAB-HPCInfra"
else
    NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-CloudInfra.git"
    echo "💻 Laptop/workstation environment detected - using NGIAB-CloudInfra"
fi

# Clone the appropriate repository
echo "📥 Cloning $NGIAB_REPO..."
cd ..
rm -rf ngiab  # Remove if exists
git clone "$NGIAB_REPO" ngiab

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to clone NGIAB repository"
    exit 1
fi

cd ngiab

# Make guide.sh executable if it exists
if [ -f "guide.sh" ]; then
    chmod +x guide.sh
    echo "✅ guide.sh found and made executable"
    
    # Test if guide.sh runs (don't actually execute it fully, just check syntax)
    echo "🧪 Testing guide.sh..."
    bash -n guide.sh
    if [ $? -eq 0 ]; then
        echo "✅ guide.sh syntax is valid"
    else
        echo "⚠️  Warning: guide.sh may have syntax issues"
    fi
else
    echo "⚠️  Warning: guide.sh not found in repository"
    echo "   Available files:"
    ls -la | head -20
fi

echo ""
echo "✅ NGIAB installation complete!"
echo "📍 Installation directory: $(pwd)"
echo "🚀 To run NGIAB setup, execute: ./guide.sh"
if [ "$IS_HPC" = "true" ]; then
    echo "ℹ️  Note: You're on an HPC system - make sure to run guide.sh on a compute node"
fi
'''
],
            'dependencies': [],
            'test_command': None,  # Can't really test without running the full guide
            'verify_install': {
                'file_paths': ['guide.sh'],
                'check_type': 'exists'
            },
            'order': 9
        }
    }

    def _check_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check which dependencies are missing from the system."""
        missing_deps = []
        
        for dep in dependencies:
            if not shutil.which(dep):
                missing_deps.append(dep)
        
        return missing_deps

    def _verify_installation(self, tool_name: str, tool_info: Dict[str, Any], 
                        install_dir: Path) -> bool:
        """Verify that a tool was installed correctly."""
        try:
            # Look for expected executable or library
            exe_name = tool_info['default_exe']
            
            # Common locations to check
            possible_paths = [
                install_dir / exe_name,
                install_dir / 'bin' / exe_name,
                install_dir / 'build' / exe_name,
                install_dir / 'route' / 'bin' / exe_name,  # mizuRoute specific
                install_dir / exe_name.replace('.exe', ''),  # Linux version
                install_dir / 'install' / 'sundials' / exe_name,  # SUNDIALS specific
            ]
            
            for exe_path in possible_paths:
                if exe_path.exists():
                    print(f"   ✅ Executable/library found: {exe_path}")
                    return True
            
            print(f"   ⚠️  Executable/library not found in expected locations")
            print(f"      Searched for: {exe_name}")
            print(f"      In directory: {install_dir}")
            
            # Show what actually exists
            if install_dir.exists():
                print(f"      Directory contents:")
                for item in sorted(install_dir.iterdir())[:10]:
                    print(f"         {item.name}")
            
            return False
            
        except Exception as e:
            print(f"   ⚠️  Verification error: {str(e)}")
            return False

    def get_executables(self, specific_tools: List[str] = None, confluence_instance=None, 
                    force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clone and install external tool repositories with dependency resolution.
        
        Args:
            specific_tools: List of specific tools to install, or None for all
            confluence_instance: CONFLUENCE instance for config access
            force: Force reinstallation even if tools exist
            dry_run: Show what would be done without actually doing it
            
        Returns:
            Dictionary with installation results
        """
        print(f"\n🚀 {'Planning' if dry_run else 'Installing'} External Tools:")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN - No actual installation will occur")
            print("-" * 30)
        
        installation_results = {
            'successful': [],
            'failed': [],
            'skipped': [],
            'errors': [],
            'dry_run': dry_run
        }
        
        # Get config
        config = {}
        if confluence_instance and hasattr(confluence_instance, 'config'):
            config = confluence_instance.config
        else:
            try:
                config_path = Path('./0_config_files/config_template.yaml')                
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check if paths exist and fix them if needed
                    config = self._ensure_valid_config_paths(config, config_path)
            except:
                pass
        
        # Determine installation directory
        data_dir = config.get('CONFLUENCE_DATA_DIR', '.')
        install_base_dir = Path(data_dir) / 'installs'
        
        print(f"📁 Installation directory: {install_base_dir}")
        
        if not dry_run:
            install_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which tools to install
        if specific_tools is None:
            tools_to_install = list(self.external_tools.keys())
        else:
            tools_to_install = []
            for tool in specific_tools:
                if tool in self.external_tools:
                    tools_to_install.append(tool)
                else:
                    print(f"⚠️  Unknown tool: {tool}")
                    installation_results['errors'].append(f"Unknown tool: {tool}")
        
        # Resolve dependencies and sort by install order
        tools_to_install = self._resolve_tool_dependencies(tools_to_install)
        
        print(f"🎯 Installing tools in order: {', '.join(tools_to_install)}")
        
        # Install each tool
        for tool_name in tools_to_install:
            tool_info = self.external_tools[tool_name]
            print(f"\n🔧 {'Planning' if dry_run else 'Installing'} {tool_name.upper()}:")
            print(f"   📝 {tool_info['description']}")
            
            tool_install_dir = install_base_dir / tool_info['install_dir']
            repository_url = tool_info['repository']
            branch = tool_info.get('branch')
            
            try:
                # Check if already exists
                if tool_install_dir.exists() and not force:
                    print(f"   ⏭️  Skipping - already exists at: {tool_install_dir}")
                    print(f"   💡 Use --force_install to reinstall")
                    installation_results['skipped'].append(tool_name)
                    continue
                
                if dry_run:
                    print(f"   🔍 Would clone: {repository_url}")
                    if branch:
                        print(f"   🌿 Would checkout branch: {branch}")
                    print(f"   📂 Target directory: {tool_install_dir}")
                    print(f"   🔨 Would run build commands:")
                    for cmd in tool_info['build_commands']:
                        print(f"      {cmd[:100]}...")
                    installation_results['successful'].append(f"{tool_name} (dry run)")
                    continue
                
                # Remove existing if force reinstall
                if tool_install_dir.exists() and force:
                    print(f"   🗑️  Removing existing installation: {tool_install_dir}")
                    shutil.rmtree(tool_install_dir)
                
                # Clone repository or create directory for non-git tools
                if repository_url:
                    # Git-based installation
                    print(f"   📥 Cloning from: {repository_url}")
                    if branch:
                        print(f"   🌿 Checking out branch: {branch}")
                        clone_cmd = ['git', 'clone', '-b', branch, repository_url, str(tool_install_dir)]
                    else:
                        clone_cmd = ['git', 'clone', repository_url, str(tool_install_dir)]
                    
                    clone_result = subprocess.run(
                        clone_cmd,
                        capture_output=True, text=True, check=True
                    )
                    
                    print(f"   ✅ Clone successful")
                else:
                    # Non-git installation (e.g., wget-based like SUNDIALS)
                    print(f"   📂 Creating installation directory")
                    tool_install_dir.mkdir(parents=True, exist_ok=True)
                    print(f"   ✅ Directory created: {tool_install_dir}")
                
                # Check dependencies
                missing_deps = self._check_dependencies(tool_info['dependencies'])
                if missing_deps:
                    print(f"   ⚠️  Missing system dependencies: {', '.join(missing_deps)}")
                    print(f"   💡 These may be available as modules - check with 'module avail'")
                    installation_results['errors'].append(f"{tool_name}: missing system dependencies {missing_deps}")
                
                # Check if tool dependencies (requires) are satisfied
                if 'requires' in tool_info:
                    required_tools = tool_info['requires']
                    for req_tool in required_tools:
                        req_tool_dir = install_base_dir / self.external_tools[req_tool]['install_dir']
                        if not req_tool_dir.exists():
                            error_msg = f"{tool_name} requires {req_tool} but it's not installed"
                            print(f"   ❌ {error_msg}")
                            installation_results['errors'].append(error_msg)
                            installation_results['failed'].append(tool_name)
                            continue
                
                # Run build commands
                if tool_info['build_commands']:
                    print(f"   🔨 Running build commands...")
                    
                    original_dir = os.getcwd()
                    os.chdir(tool_install_dir)
                    
                    try:
                        for i, cmd in enumerate(tool_info['build_commands'], 1):
                            print(f"      Building step {i}...")
                            # Run as shell script to handle multi-line commands
                            build_result = subprocess.run(
                                cmd, 
                                shell=True, 
                                check=True, 
                                capture_output=True, 
                                text=True,
                                executable='/bin/bash'
                            )
                            # For important builds, show more output
                            if tool_name in ['summa', 'sundials', 'mizuroute']:
                                # Show all output for critical tools
                                if build_result.stdout:
                                    print(f"         === Build Output ===")
                                    for line in build_result.stdout.strip().split('\n'):
                                        print(f"         {line}")
                            else:
                                # Show last few lines for other tools
                                if build_result.stdout:
                                    lines = build_result.stdout.strip().split('\n')
                                    for line in lines[-5:]:
                                        print(f"         {line}")
                            print(f"         ✅ Build step {i} completed")
                        
                        print(f"   ✅ Build successful")
                        installation_results['successful'].append(tool_name)
                        
                    except subprocess.CalledProcessError as build_error:
                        print(f"   ❌ Build failed: {build_error}")
                        if build_error.stdout:
                            print(f"      === Build Output ===")
                            for line in build_error.stdout.strip().split('\n'):
                                print(f"         {line}")
                        if build_error.stderr:
                            print(f"      === Error Output ===")
                            for line in build_error.stderr.strip().split('\n'):
                                print(f"         {line}")
                        installation_results['failed'].append(tool_name)
                        installation_results['errors'].append(f"{tool_name} build failed")
                    
                    finally:
                        os.chdir(original_dir)
                else:
                    print(f"   ✅ No build required")
                    installation_results['successful'].append(tool_name)
                
                # Verify installation
                self._verify_installation(tool_name, tool_info, tool_install_dir)
                
            except subprocess.CalledProcessError as e:
                if repository_url:
                    error_msg = f"Failed to clone {repository_url}: {e.stderr if e.stderr else str(e)}"
                else:
                    error_msg = f"Failed during installation: {e.stderr if e.stderr else str(e)}"
                print(f"   ❌ {error_msg}")
                installation_results['failed'].append(tool_name)
                installation_results['errors'].append(f"{tool_name}: {error_msg}")
            
            except Exception as e:
                error_msg = f"Installation error: {str(e)}"
                print(f"   ❌ {error_msg}")
                installation_results['failed'].append(tool_name)
                installation_results['errors'].append(f"{tool_name}: {error_msg}")
        
        # Print summary
        self._print_installation_summary(installation_results, dry_run)
        
        return installation_results

    def _ensure_valid_config_paths(self, config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
        """
        Ensure CONFLUENCE_DATA_DIR and CONFLUENCE_CODE_DIR paths exist and are valid.
        If they don't exist or are inaccessible, update them to sensible defaults.
        
        Args:
            config: Configuration dictionary
            config_path: Path to the config file
            
        Returns:
            Updated configuration dictionary
        """
        import os
        
        data_dir = config.get('CONFLUENCE_DATA_DIR')
        code_dir = config.get('CONFLUENCE_CODE_DIR')
        
        data_dir_valid = False
        code_dir_valid = False
        
        # Check if CONFLUENCE_DATA_DIR exists and is accessible
        if data_dir:
            try:
                data_path = Path(data_dir)
                # Try to check if path exists and is writable
                if data_path.exists():
                    # Test write access
                    test_file = data_path / '.confluence_test'
                    try:
                        test_file.touch()
                        test_file.unlink()
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
                else:
                    # Try to create the directory
                    try:
                        data_path.mkdir(parents=True, exist_ok=True)
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
            except Exception:
                pass
        
        # Check if CONFLUENCE_CODE_DIR exists and is accessible
        if code_dir:
            try:
                code_path = Path(code_dir)
                if code_path.exists() and os.access(code_path, os.R_OK):
                    code_dir_valid = True
            except Exception:
                pass
        
        # If either path is invalid, update config
        if not data_dir_valid or not code_dir_valid:
            print(f"\n⚠️  Detected invalid or inaccessible paths in config template:")
            
            if not code_dir_valid:
                print(f"   📂 CONFLUENCE_CODE_DIR: {code_dir} (inaccessible)")
            
            if not data_dir_valid:
                print(f"   📂 CONFLUENCE_DATA_DIR: {data_dir} (inaccessible)")
            
            print(f"\n🔧 Auto-configuring paths for fresh installation...")
            
            # Set CONFLUENCE_CODE_DIR to current directory
            if not code_dir_valid:
                new_code_dir = Path.cwd().resolve()
                config['CONFLUENCE_CODE_DIR'] = str(new_code_dir)
                print(f"   ✅ CONFLUENCE_CODE_DIR set to: {new_code_dir}")
            
            # Set CONFLUENCE_DATA_DIR to ../CONFLUENCE_data
            if not data_dir_valid:
                new_data_dir = (Path.cwd().parent / 'CONFLUENCE_data').resolve()
                config['CONFLUENCE_DATA_DIR'] = str(new_data_dir)
                
                # Create the data directory if it doesn't exist
                try:
                    new_data_dir.mkdir(parents=True, exist_ok=True)
                    print(f"   ✅ CONFLUENCE_DATA_DIR set to: {new_data_dir}")
                    print(f"   📁 Created data directory")
                except Exception as e:
                    print(f"   ⚠️  Could not create data directory: {e}")
            
            # Save updated config back to file
            try:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"   💾 Updated config template saved to: {config_path}")
            except Exception as e:
                print(f"   ⚠️  Could not save updated config: {e}")
        
        return config

    def _resolve_tool_dependencies(self, tools: List[str]) -> List[str]:
        """
        Resolve dependencies between tools and return sorted list.
        
        Args:
            tools: List of tool names to install
            
        Returns:
            Sorted list of tools with dependencies resolved
        """
        # Build dependency graph
        tools_with_deps = set(tools)
        
        # Add required dependencies
        for tool in tools:
            if tool in self.external_tools and 'requires' in self.external_tools[tool]:
                required = self.external_tools[tool]['requires']
                tools_with_deps.update(required)
        
        # Sort by order field, then alphabetically
        sorted_tools = sorted(
            tools_with_deps,
            key=lambda t: (
                self.external_tools.get(t, {}).get('order', 999),
                t
            )
        )
        
        return sorted_tools


    def _print_installation_summary(self, results: Dict[str, Any], dry_run: bool) -> None:
        """Print installation summary."""
        successful_count = len(results['successful'])
        failed_count = len(results['failed'])
        skipped_count = len(results['skipped'])
        
        print(f"\n📊 Installation Summary:")
        if dry_run:
            print(f"   🔍 Would install: {successful_count} tools")
            print(f"   ⏭️  Would skip: {skipped_count} tools")
        else:
            print(f"   ✅ Successful: {successful_count} tools")
            print(f"   ❌ Failed: {failed_count} tools")
            print(f"   ⏭️  Skipped: {skipped_count} tools")
        
        if results['errors']:
            print(f"\n❌ Errors encountered:")
            for error in results['errors']:
                print(f"   • {error}")
        
        if successful_count > 0 and not dry_run:
            print(f"\n✅ Installation complete!")
            print(f"💡 Run 'python CONFLUENCE.py --validate_binaries' to verify")

    def handle_binary_management(cli_manager, execution_plan, confluence_instance=None):
        """
        Handle binary management operations (validate_binaries, get_executables).
        
        Args:
            cli_manager: CLIArgumentManager instance
            execution_plan: Execution plan from CLI manager
            confluence_instance: CONFLUENCE instance (optional)
        
        Returns:
            bool: Success status
        """
        try:
            binary_ops = execution_plan.get('binary_operations', {})
            
            # Handle --validate_binaries
            if binary_ops.get('validate_binaries', False):
                print("\n🔧 Binary Validation Requested")
                results = cli_manager.validate_binaries(confluence_instance)
                return len(results['missing_tools']) == 0 and len(results['failed_tools']) == 0
            
            # Handle --get_executables
            if binary_ops.get('get_executables') is not None:
                print("\n🚀 Executable Installation Requested")
                specific_tools = binary_ops.get('get_executables')
                force_install = binary_ops.get('force_install', False)
                dry_run = execution_plan.get('settings', {}).get('dry_run', False)
                
                # If empty list, install all tools
                if isinstance(specific_tools, list) and len(specific_tools) == 0:
                    specific_tools = None
                
                results = cli_manager.get_executables(
                    specific_tools=specific_tools,
                    confluence_instance=confluence_instance,
                    force=force_install,
                    dry_run=dry_run
                )
                
                return len(results['failed_tools']) == 0
            
            return True
            
        except Exception as e:
            print(f"❌ Error in binary management: {str(e)}")
            return False


    def validate_binaries(self, confluence_instance=None) -> Dict[str, Any]:
        """
        Validate that required binary executables exist and are functional.
        
        Args:
            confluence_instance: CONFLUENCE system instance for accessing config
            
        Returns:
            Dictionary with validation results for each tool
        """
        print("\n🔧 Validating External Tool Binaries:")
        print("=" * 50)
        
        validation_results = {
            'valid_tools': [],
            'missing_tools': [],
            'failed_tools': [],
            'warnings': [],
            'summary': {}
        }
        
        # Get config if available
        config = {}
        if confluence_instance and hasattr(confluence_instance, 'config'):
            config = confluence_instance.config
        elif confluence_instance and hasattr(confluence_instance, 'workflow_orchestrator'):
            config = confluence_instance.workflow_orchestrator.config
        
        # If no config available, try to load default
        if not config:
            try:
                config_path = Path('./0_config_files/config_template.yaml')
                
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    print(f"📄 Using config from: {config_path}")
                else:
                    print("⚠️  No configuration file found - using default paths")
            except Exception as e:
                print(f"⚠️  Could not load config: {str(e)} - using default paths")
        
        # Validate each tool
        for tool_name, tool_info in self.external_tools.items():
            print(f"\n🔍 Checking {tool_name.upper()}:")
            
            tool_result = {
                'name': tool_name,
                'description': tool_info['description'],
                'status': 'unknown',
                'path': None,
                'executable': None,
                'version': None,
                'errors': []
            }
            
            try:
                # Determine tool path
                config_path_key = tool_info['config_path_key']
                tool_path = config.get(config_path_key, 'default')
                
                if tool_path == 'default':
                    # Use default path construction
                    data_dir = config.get('CONFLUENCE_DATA_DIR', '.')
                    tool_path = Path(data_dir) / tool_info['default_path_suffix']
                else:
                    tool_path = Path(tool_path)
                
                tool_result['path'] = str(tool_path)
                
                # Determine executable name
                config_exe_key = tool_info.get('config_exe_key')
                if config_exe_key and config_exe_key in config:
                    exe_name = config[config_exe_key]
                else:
                    exe_name = tool_info['default_exe']
                
                tool_result['executable'] = exe_name
                
                # Check if executable exists
                exe_path = tool_path / exe_name
                if not exe_path.exists():
                    # Try without extension for Linux systems
                    exe_name_no_ext = exe_name.replace('.exe', '')
                    exe_path_no_ext = tool_path / exe_name_no_ext
                    if exe_path_no_ext.exists():
                        exe_path = exe_path_no_ext
                        tool_result['executable'] = exe_name_no_ext
                
                if exe_path.exists():
                    # Check if tool needs execution test
                    test_cmd = tool_info.get('test_command')
                    
                    # If no test command, just verify existence (for libraries, scripts, etc.)
                    if test_cmd is None:
                        tool_result['status'] = 'valid'
                        tool_result['version'] = 'Installed (existence verified)'
                        validation_results['valid_tools'].append(tool_name)
                        print(f"   ✅ Found at: {exe_path}")
                        print(f"   ✅ Status: Installed")
                    else:
                        # Test by executing
                        try:
                            result = subprocess.run(
                                [str(exe_path), test_cmd],
                                capture_output=True, 
                                text=True, 
                                timeout=10
                            )
                            if result.returncode == 0 or test_cmd == '--help':
                                tool_result['status'] = 'valid'
                                tool_result['version'] = result.stdout.strip()[:100] if result.stdout else 'Available'
                                validation_results['valid_tools'].append(tool_name)
                                print(f"   ✅ Found at: {exe_path}")
                                print(f"   ✅ Status: Working")
                            else:
                                tool_result['status'] = 'failed'
                                tool_result['errors'].append(f"Test command failed: {result.stderr}")
                                validation_results['failed_tools'].append(tool_name)
                                print(f"   🟡 Found but test failed: {exe_path}")
                                print(f"   ⚠️  Error: {result.stderr[:100]}")
                        
                        except subprocess.TimeoutExpired:
                            tool_result['status'] = 'timeout'
                            tool_result['errors'].append("Test command timed out")
                            validation_results['warnings'].append(f"{tool_name}: test timed out")
                            print(f"   🟡 Found but test timed out: {exe_path}")
                        
                        except Exception as test_error:
                            tool_result['status'] = 'test_error'
                            tool_result['errors'].append(f"Test error: {str(test_error)}")
                            validation_results['warnings'].append(f"{tool_name}: {str(test_error)}")
                            print(f"   🟡 Found but couldn't test: {exe_path}")
                            print(f"   ⚠️  Test error: {str(test_error)}")
                
                else:
                    tool_result['status'] = 'missing'
                    tool_result['errors'].append(f"Executable not found at: {exe_path}")
                    validation_results['missing_tools'].append(tool_name)
                    print(f"   ❌ Not found: {exe_path}")
                    print(f"   💡 Try: python CONFLUENCE.py --get_executables {tool_name}")
            
            except Exception as e:
                tool_result['status'] = 'error'
                tool_result['errors'].append(f"Validation error: {str(e)}")
                validation_results['failed_tools'].append(tool_name)
                print(f"   ❌ Validation error: {str(e)}")
            
            validation_results['summary'][tool_name] = tool_result
        
        # Print summary
        total_tools = len(self.external_tools)
        valid_count = len(validation_results['valid_tools'])
        missing_count = len(validation_results['missing_tools'])
        failed_count = len(validation_results['failed_tools'])
        
        print(f"\n📊 Binary Validation Summary:")
        print(f"   ✅ Valid: {valid_count}/{total_tools}")
        print(f"   ❌ Missing: {missing_count}/{total_tools}")
        print(f"   🔧 Failed: {failed_count}/{total_tools}")
        
        if missing_count > 0:
            print(f"\n💡 To install missing tools:")
            print(f"   python CONFLUENCE.py --get_executables")
            print(f"   python CONFLUENCE.py --get_executables {' '.join(validation_results['missing_tools'])}")
        
        if failed_count > 0:
            print(f"\n🔧 Failed tools may need rebuilding or dependency installation")
        
        return validation_results

    def validate_step_availability(self, confluence_instance, requested_steps: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate that requested steps are available in the current workflow.
        
        Args:
            confluence_instance: CONFLUENCE system instance
            requested_steps: List of step names requested by user
            
        Returns:
            Tuple of (available_steps, unavailable_steps)
        """
        available_steps = []
        unavailable_steps = []
        
        if hasattr(confluence_instance, 'workflow_orchestrator'):
            # Get actual workflow steps
            workflow_steps = confluence_instance.workflow_orchestrator.define_workflow_steps()
            actual_functions = {step_func.__name__ for step_func, _ in workflow_steps}
            
            # Check each requested step
            for step_name in requested_steps:
                if step_name in self.workflow_steps:
                    function_name = self.workflow_steps[step_name].get('function_name', step_name)
                    if function_name in actual_functions:
                        available_steps.append(step_name)
                    else:
                        unavailable_steps.append(step_name)
                else:
                    unavailable_steps.append(step_name)
        else:
            # Fallback - assume all defined steps are available
            for step_name in requested_steps:
                if step_name in self.workflow_steps:
                    available_steps.append(step_name)
                else:
                    unavailable_steps.append(step_name)
        
        return available_steps, unavailable_steps

    def resume_workflow_from_step(self, step_name: str, confluence_instance) -> List[str]:
        """
        Determine which steps to execute when resuming from a specific step.
        
        Args:
            step_name: Name of the step to resume from
            confluence_instance: CONFLUENCE system instance
            
        Returns:
            List of step names to execute
        """
        print(f"\n🔄 Resuming Workflow from: {step_name}")
        print("=" * 50)
        
        try:
            # Get workflow definition
            workflow_steps = confluence_instance.workflow_orchestrator.define_workflow_steps()
            
            # Create mapping from function names to steps
            step_functions = {}
            step_order = []
            
            for i, (step_func, check_func, description) in enumerate(workflow_steps):
                func_name = step_func.__name__
                step_functions[func_name] = {
                    'function': step_func,
                    'check': check_func,
                    'description': description,
                    'order': i
                }
                step_order.append(func_name)
            
            # Map CLI step names to function names
            cli_to_function_map = {
                'setup_project': 'setup_project',
                'create_pour_point': 'create_pour_point',
                'acquire_attributes': 'acquire_attributes',
                'define_domain': 'define_domain',
                'discretize_domain': 'discretize_domain',
                'process_observed_data': 'process_observed_data',
                'acquire_forcings': 'acquire_forcings',
                'run_model_agnostic_preprocessing': 'run_model_agnostic_preprocessing',
                'setup_model': 'preprocess_models',
                'run_model': 'run_models',
                'calibrate_model': 'calibrate_model',
                'run_emulation': 'run_emulation',
                'run_benchmarking': 'run_benchmarking',
                'run_decision_analysis': 'run_decision_analysis',
                'run_sensitivity_analysis': 'run_sensitivity_analysis',
                'postprocess_results': 'postprocess_results'
            }
            
            # Find the function name for the CLI step
            function_name = cli_to_function_map.get(step_name, step_name)
            
            if function_name not in step_functions:
                print(f"❌ Step '{step_name}' not found in workflow")
                print(f"Available steps: {', '.join(cli_to_function_map.keys())}")
                return []
            
            # Find the starting position
            start_index = step_functions[function_name]['order']
            
            # Get steps to execute (from start_index onwards)
            steps_to_execute = []
            skipped_steps = []
            
            for i, func_name in enumerate(step_order):
                cli_name = None
                # Find CLI name for this function
                for cli_step, func_step in cli_to_function_map.items():
                    if func_step == func_name:
                        cli_name = cli_step
                        break
                
                if i < start_index:
                    # Check if earlier steps are completed
                    step_info = step_functions[func_name]
                    try:
                        is_completed = step_info['check']() if callable(step_info['check']) else False
                        if is_completed:
                            skipped_steps.append({
                                'name': cli_name or func_name,
                                'status': 'completed',
                                'description': step_info['description']
                            })
                        else:
                            skipped_steps.append({
                                'name': cli_name or func_name,
                                'status': 'missing_dependencies',
                                'description': step_info['description']
                            })
                    except:
                        skipped_steps.append({
                            'name': cli_name or func_name,
                            'status': 'unknown',
                            'description': step_info['description']
                        })
                else:
                    if cli_name:
                        steps_to_execute.append(cli_name)
            
            # Print resume plan
            print(f"📍 Starting from step {start_index + 1}: {step_name}")
            print(f"🎯 Will execute {len(steps_to_execute)} steps")
            
            if skipped_steps:
                print(f"\n⏭️  Skipped Steps (executed previously):")
                for i, step in enumerate(skipped_steps, 1):
                    status_icon = {
                        'completed': '✅',
                        'missing_dependencies': '⚠️ ',
                        'unknown': '❓'
                    }.get(step['status'], '❓')
                    
                    print(f"   {i:2d}. {status_icon} {step['name']} - {step['description']}")
                    
                    if step['status'] == 'missing_dependencies':
                        print(f"       ⚠️  Warning: Dependencies may be missing")
            
            if steps_to_execute:
                print(f"\n🚀 Steps to Execute:")
                for i, step in enumerate(steps_to_execute, start_index + 1):
                    step_description = step_functions.get(
                        cli_to_function_map.get(step, step), {}
                    ).get('description', step)
                    print(f"   {i:2d}. ⏳ {step} - {step_description}")
            
            # Check for potential issues
            warnings = []
            missing_deps = [s for s in skipped_steps if s['status'] == 'missing_dependencies']
            if missing_deps:
                warnings.append(f"⚠️  {len(missing_deps)} earlier steps appear incomplete")
            
            if warnings:
                print(f"\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"   {warning}")
                print(f"   Consider running --workflow_status to check dependencies")
            
            print(f"\n💡 Resume command: python CONFLUENCE.py {' --'.join([''] + steps_to_execute)}")
            
            return steps_to_execute
            
        except Exception as e:
            print(f"❌ Error planning workflow resume: {str(e)}")
            return []

    def clean_workflow_files(self, clean_level: str = 'intermediate', confluence_instance = None, 
                            dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean workflow files and directories based on the specified level.
        
        Args:
            clean_level: Level of cleaning ('intermediate', 'outputs', 'all')
            confluence_instance: CONFLUENCE system instance
            dry_run: If True, show what would be cleaned without actually deleting
            
        Returns:
            Dictionary with cleaning results
        """
        print(f"\n🧹 Cleaning Workflow Files (Level: {clean_level})")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN - No files will be deleted")
            print("-" * 30)
        
        cleaning_results = {
            'level': clean_level,
            'dry_run': dry_run,
            'files_removed': [],
            'directories_removed': [],
            'errors': [],
            'total_size_freed': 0,
            'summary': {}
        }
        
        if not confluence_instance:
            print("❌ Cannot clean files without CONFLUENCE instance")
            return cleaning_results
        
        try:
            project_dir = confluence_instance.workflow_orchestrator.project_dir
            
            if not project_dir.exists():
                print(f"📁 Project directory not found: {project_dir}")
                return cleaning_results
            
            print(f"📁 Project directory: {project_dir}")
            
            # Define what to clean at each level
            clean_targets = self._get_clean_targets(project_dir, clean_level)
            
            # Show what will be cleaned
            print(f"\n🎯 Targets for cleaning:")
            for category, paths in clean_targets.items():
                if paths:
                    print(f"   📂 {category.title()}:")
                    for path in paths:
                        if path.exists():
                            size = self._get_path_size(path)
                            size_str = f" ({size:.1f} MB)" if size > 0 else ""
                            print(f"      🗂️  {path.relative_to(project_dir)}{size_str}")
                        else:
                            print(f"      ❓ {path.relative_to(project_dir)} (not found)")
            
            if not any(clean_targets.values()):
                print("✨ No files found to clean!")
                return cleaning_results
            
            # Confirm deletion (unless dry run)
            if not dry_run:
                total_size = sum(self._get_path_size(path) 
                            for paths in clean_targets.values() 
                            for path in paths if path.exists())
                
                print(f"\n💾 Total space to be freed: {total_size:.1f} MB")
                confirm = input("⚠️  Proceed with deletion? (y/N): ").strip().lower()
                
                if confirm not in ['y', 'yes']:
                    print("❌ Cleaning cancelled by user")
                    return cleaning_results
            
            # Perform cleaning
            print(f"\n🧹 {'Would clean' if dry_run else 'Cleaning'} files...")
            
            for category, paths in clean_targets.items():
                category_results = {
                    'files_removed': 0,
                    'directories_removed': 0,
                    'size_freed': 0,
                    'errors': 0
                }
                
                for path in paths:
                    if not path.exists():
                        continue
                    
                    try:
                        size = self._get_path_size(path)
                        
                        if not dry_run:
                            if path.is_file():
                                path.unlink()
                                cleaning_results['files_removed'].append(str(path))
                                category_results['files_removed'] += 1
                            elif path.is_dir():
                                import shutil
                                shutil.rmtree(path)
                                cleaning_results['directories_removed'].append(str(path))
                                category_results['directories_removed'] += 1
                        else:
                            # Dry run - just record what would be removed
                            if path.is_file():
                                cleaning_results['files_removed'].append(str(path))
                                category_results['files_removed'] += 1
                            elif path.is_dir():
                                cleaning_results['directories_removed'].append(str(path))
                                category_results['directories_removed'] += 1
                        
                        category_results['size_freed'] += size
                        cleaning_results['total_size_freed'] += size
                        
                        action = "Would remove" if dry_run else "Removed"
                        print(f"   {'🗑️ ' if not dry_run else '👁️ '} {action}: {path.relative_to(project_dir)} ({size:.1f} MB)")
                        
                    except Exception as e:
                        error_msg = f"Error cleaning {path}: {str(e)}"
                        cleaning_results['errors'].append(error_msg)
                        category_results['errors'] += 1
                        print(f"   ❌ {error_msg}")
                
                cleaning_results['summary'][category] = category_results
            
            # Print summary
            total_files = len(cleaning_results['files_removed'])
            total_dirs = len(cleaning_results['directories_removed'])
            total_size = cleaning_results['total_size_freed']
            total_errors = len(cleaning_results['errors'])
            
            print(f"\n📊 Cleaning Summary:")
            action = "Would clean" if dry_run else "Cleaned"
            print(f"   {action}: {total_files} files, {total_dirs} directories")
            print(f"   💾 Space {'that would be' if dry_run else ''} freed: {total_size:.1f} MB")
            
            if total_errors > 0:
                print(f"   ❌ Errors: {total_errors}")
            
            if not dry_run:
                print("   ✅ Cleaning completed successfully!")
            else:
                print("   👁️  Dry run completed - use without --dry_run to actually clean")
            
        except Exception as e:
            error_msg = f"Error during cleaning: {str(e)}"
            cleaning_results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
        
        return cleaning_results


    def _detect_environment(self) -> str:
        """
        Detect whether we're running on HPC or a personal computer.
        
        Detection logic:
        - Check for common HPC schedulers (SLURM, PBS, SGE)
        - Check for HPC-specific environment variables
        - Default to 'laptop' if no HPC indicators found
        
        Returns:
            'hpc' or 'laptop'
        """
        # Check for HPC scheduler commands
        hpc_schedulers = ['sbatch', 'qsub', 'bsub']
        for scheduler in hpc_schedulers:
            if shutil.which(scheduler):
                return 'hpc'
        
        # Check for common HPC environment variables
        hpc_env_vars = [
            'SLURM_CLUSTER_NAME',
            'SLURM_JOB_ID',
            'PBS_JOBID',
            'SGE_CLUSTER_NAME',
            'LOADL_STEP_ID'
        ]
        
        for env_var in hpc_env_vars:
            if env_var in os.environ:
                return 'hpc'
        
        # Check for /scratch directory (common on HPC systems)
        if Path('/scratch').exists():
            return 'hpc'
        
        return 'laptop'


    def _get_clean_targets(self, project_dir: Path, clean_level: str) -> Dict[str, List[Path]]:
        """Get list of files/directories to clean based on level."""
        targets = {
            'intermediate': [],
            'logs': [],
            'outputs': [],
            'all': []
        }
        
        # Intermediate files (temporary processing files)
        intermediate_patterns = [
            'tmp*', '*.tmp', '*.temp', 
            '*_temp*', '*_intermediate*',
            '*/cache/*', '*/temp/*'
        ]
        
        # Log files
        log_patterns = [
            '*.log', 'logs/*', '*_log.txt',
            'debug_*', 'error_*'
        ]
        
        # Output files (results that can be regenerated)
        output_dirs = [
            'simulations/*', 
            'optimisation/*',
            'plots/*'
        ]
        
        if clean_level in ['intermediate', 'all']:
            # Find intermediate files
            for pattern in intermediate_patterns:
                targets['intermediate'].extend(project_dir.glob(pattern))
                targets['intermediate'].extend(project_dir.rglob(pattern))
        
        if clean_level in ['intermediate', 'outputs', 'all']:
            # Find log files
            for pattern in log_patterns:
                targets['logs'].extend(project_dir.glob(pattern))
                targets['logs'].extend(project_dir.rglob(pattern))
        
        if clean_level in ['outputs', 'all']:
            # Find output directories
            for pattern in output_dirs:
                targets['outputs'].extend(project_dir.glob(pattern))
        
        if clean_level == 'all':
            # Add specific directories for complete cleaning
            all_clean_dirs = [
                project_dir / 'simulations' / 'temp',
                project_dir / 'optimisation' / 'temp',
                project_dir / 'forcing' / 'temp',
                project_dir / 'plots' / 'temp'
            ]
            targets['all'].extend([d for d in all_clean_dirs if d.exists()])
        
        # Remove duplicates and sort
        for key in targets:
            targets[key] = sorted(list(set(targets[key])))
        
        return targets

    def _get_path_size(self, path: Path) -> float:
        """Get size of file or directory in MB."""
        if not path.exists():
            return 0.0
        
        try:
            if path.is_file():
                return path.stat().st_size / (1024 * 1024)  # Convert to MB
            elif path.is_dir():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
        
        return 0.0

    def get_detailed_workflow_status(self, confluence_instance = None) -> Dict[str, Any]:
        """
        Get detailed workflow status with step completion, file checks, and timestamps.
        
        Args:
            confluence_instance: CONFLUENCE system instance
            
        Returns:
            Dictionary with detailed status information
        """
        from datetime import datetime
        import os
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'steps': {},
            'summary': {
                'total_steps': 0,
                'completed': 0,
                'failed': 0,
                'pending': 0
            },
            'config_validation': {},
            'system_info': {}
        }
        
        if not confluence_instance:
            return status
        
        try:
            # Get workflow definition
            workflow_steps = confluence_instance.workflow_orchestrator.define_workflow_steps()
            status['summary']['total_steps'] = len(workflow_steps)
            
            project_dir = confluence_instance.workflow_orchestrator.project_dir
            
            # Check each workflow step
            for i, (step_func, check_func, description) in enumerate(workflow_steps, 1):
                step_name = step_func.__name__
                
                step_status = {
                    'name': step_name,
                    'description': description,
                    'order': i,
                    'status': 'pending',
                    'files_exist': False,
                    'file_checks': {},
                    'timestamps': {},
                    'size_info': {}
                }
                
                # Check if step outputs exist
                try:
                    files_exist = check_func() if callable(check_func) else False
                    step_status['files_exist'] = files_exist
                    
                    if files_exist:
                        step_status['status'] = 'completed'
                        status['summary']['completed'] += 1
                    else:
                        step_status['status'] = 'pending'
                        status['summary']['pending'] += 1
                        
                except Exception as e:
                    step_status['status'] = 'failed'
                    step_status['error'] = str(e)
                    status['summary']['failed'] += 1
                
                # Get detailed file information for completed steps
                if step_status['files_exist']:
                    step_status.update(self._get_step_file_details(step_name, project_dir))
                
                status['steps'][step_name] = step_status
            
            # Config validation
            status['config_validation'] = self._validate_config_status(confluence_instance)
            
            # System info
            status['system_info'] = self._get_system_info()
            
        except Exception as e:
            status['error'] = f"Error getting workflow status: {str(e)}"
        
        return status

    def _get_step_file_details(self, step_name: str, project_dir: Path) -> Dict[str, Any]:
        """Get detailed file information for a workflow step."""
        file_details = {
            'file_checks': {},
            'timestamps': {},
            'size_info': {}
        }
        
        # Define expected output locations for each step
        step_outputs = {
            'setup_project': [project_dir / 'shapefiles'],
            'create_pour_point': [project_dir / 'shapefiles' / 'pour_point'],
            'acquire_attributes': [project_dir / 'attributes'],
            'define_domain': [project_dir / 'shapefiles' / 'catchment'],
            'discretize_domain': [project_dir / 'shapefiles' / 'river_basins'],
            'process_observed_data': [project_dir / 'observations' / 'streamflow' / 'preprocessed'],
            'acquire_forcings': [project_dir / 'forcing' / 'raw_data'],
            'run_model_agnostic_preprocessing': [project_dir / 'forcing' / 'basin_averaged_data'],
            'preprocess_models': [project_dir / 'forcing'],
            'run_models': [project_dir / 'simulations'],
            'calibrate_model': [project_dir / 'optimisation'],
            'postprocess_results': [project_dir / 'results']
        }
        
        expected_files = step_outputs.get(step_name, [])
        
        for file_path in expected_files:
            if file_path.exists():
                file_details['file_checks'][str(file_path)] = True
                
                # Get timestamp
                mtime = file_path.stat().st_mtime
                file_details['timestamps'][str(file_path)] = datetime.fromtimestamp(mtime).isoformat()
                
                # Get size info
                if file_path.is_file():
                    size = file_path.stat().st_size
                    file_details['size_info'][str(file_path)] = f"{size / (1024**2):.2f} MB"
                elif file_path.is_dir():
                    # Count files in directory
                    file_count = len(list(file_path.rglob('*')))
                    file_details['size_info'][str(file_path)] = f"{file_count} files"
            else:
                file_details['file_checks'][str(file_path)] = False
        
        return file_details

    def _validate_config_status(self, confluence_instance) -> Dict[str, Any]:
        """Validate configuration status."""
        config_status = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'key_settings': {}
        }
        
        try:
            config = confluence_instance.config
            
            # Check required settings
            required_settings = [
                'DOMAIN_NAME', 'EXPERIMENT_ID', 'CONFLUENCE_DATA_DIR', 
                'EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END'
            ]
            
            for setting in required_settings:
                if setting in config and config[setting]:
                    config_status['key_settings'][setting] = config[setting]
                else:
                    config_status['errors'].append(f"Missing required setting: {setting}")
                    config_status['valid'] = False
            
            # Check paths exist
            if 'CONFLUENCE_DATA_DIR' in config:
                data_dir = Path(config['CONFLUENCE_DATA_DIR'])
                if not data_dir.exists():
                    config_status['warnings'].append(f"Data directory does not exist: {data_dir}")
            
            # Check date formats
            for date_field in ['EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END']:
                if date_field in config:
                    try:
                        # Try to parse the date
                        datetime.fromisoformat(str(config[date_field]).replace(' ', 'T'))
                    except:
                        config_status['warnings'].append(f"Invalid date format in {date_field}")
            
        except Exception as e:
            config_status['valid'] = False
            config_status['errors'].append(f"Config validation error: {str(e)}")
        
        return config_status

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        system_info = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'working_directory': str(Path.cwd()),
            'user': os.getenv('USER', 'unknown')
        }
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info['memory_usage'] = f"{memory.percent:.1f}%"
            system_info['available_memory_gb'] = f"{memory.available / (1024**3):.1f}"
        except ImportError:
            pass
        
        return system_info

    def print_detailed_workflow_status(self, status: Dict[str, Any]) -> None:
        """Print detailed workflow status in a user-friendly format."""
        print("\n📊 Detailed Workflow Status")
        print("=" * 60)
        
        # Summary
        summary = status['summary']
        total = summary['total_steps']
        completed = summary['completed']
        pending = summary['pending']
        failed = summary['failed']
        
        progress = (completed / total * 100) if total > 0 else 0
        
        print(f"🎯 Progress: {completed}/{total} steps ({progress:.1f}%)")
        print(f"✅ Completed: {completed}")
        print(f"⏳ Pending: {pending}")
        print(f"❌ Failed: {failed}")
        print(f"🕐 Status as of: {status['timestamp']}")
        
        # Progress bar
        bar_length = 40
        filled_length = int(bar_length * completed // total) if total > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"📈 [{bar}] {progress:.1f}%")
        
        # Step details
        print(f"\n📋 Step Details:")
        print("-" * 60)
        
        for step_name, step_info in status.get('steps', {}).items():
            order = step_info.get('order', 0)
            step_status = step_info.get('status', 'unknown')
            description = step_info.get('description', step_name)
            
            # Status icon
            if step_status == 'completed':
                icon = '✅'
            elif step_status == 'failed':
                icon = '❌'
            else:
                icon = '⏳'
            
            print(f"{order:2d}. {icon} {step_name}")
            print(f"    📝 {description}")
            print(f"    📊 Status: {step_status}")
            
            # File checks
            if step_info.get('files_exist'):
                print(f"    📁 Output files: Present")
                
                # Show timestamps for recent files
                timestamps = step_info.get('timestamps', {})
                if timestamps:
                    latest_file = max(timestamps.items(), key=lambda x: x[1])
                    print(f"    🕐 Latest: {latest_file[1]}")
            else:
                print(f"    📁 Output files: Missing")
            
            # Size info
            size_info = step_info.get('size_info', {})
            if size_info:
                for path, size in size_info.items():
                    file_name = Path(path).name
                    print(f"    📏 {file_name}: {size}")
            
            print()
        
        # Config validation
        config_status = status.get('config_validation', {})
        print(f"\n⚙️ Configuration Status:")
        if config_status.get('valid', False):
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has issues")
            
            errors = config_status.get('errors', [])
            warnings = config_status.get('warnings', [])
            
            for error in errors:
                print(f"   ❌ Error: {error}")
            for warning in warnings:
                print(f"   ⚠️  Warning: {warning}")
        
        # Key settings
        key_settings = config_status.get('key_settings', {})
        if key_settings:
            print("\n🔑 Key Settings:")
            for key, value in key_settings.items():
                print(f"   {key}: {value}")
        
        # System info
        system_info = status.get('system_info', {})
        if system_info:
            print(f"\n💻 System Information:")
            for key, value in system_info.items():
                print(f"   {key}: {value}")


    def list_templates(self) -> None:
        """List all available configuration templates."""
        print("\n📋 Available Configuration Templates:")
        print("=" * 50)
        
        # Look for templates in common locations
        template_locations = [
            Path("./0_config_files"),
            Path("../0_config_files"),
            Path("../../0_config_files"),
        ]
        
        templates_found = []
        
        for location in template_locations:
            if location.exists():
                # Look for template files
                for template_file in location.glob("config_*.yaml"):
                    if template_file.name not in [f.name for f in templates_found]:
                        templates_found.append(template_file)
                
                # Also look for the main template
                main_template = location / "config_template.yaml"
                if main_template.exists():
                    templates_found.insert(0, main_template)
        
        if not templates_found:
            print("❌ No templates found in standard locations")
            print("   Searched:", [str(loc) for loc in template_locations])
            return
        
        for i, template in enumerate(templates_found, 1):
            try:
                # Load template to get metadata
                with open(template, 'r') as f:
                    config = yaml.safe_load(f)
                
                domain_name = config.get('DOMAIN_NAME', 'Unknown')
                model_type = config.get('HYDROLOGICAL_MODEL', 'Unknown')
                description = self._get_template_description(template, config)
                
                print(f"{i}. 📄 {template.name}")
                print(f"   📍 Path: {template}")
                print(f"   🏞️  Domain: {domain_name}")
                print(f"   🔧 Model: {model_type}")
                print(f"   📝 Description: {description}")
                print()
                
            except Exception as e:
                print(f"{i}. ❌ {template.name} (Error reading: {str(e)})")
        
        print("💡 Usage:")
        print("   python CONFLUENCE.py --config path/to/template.yaml")
        print("   python CONFLUENCE.py --pour_point LAT/LON --domain_def METHOD --domain_name NAME")

    def _get_template_description(self, template_path: Path, config: Dict[str, Any]) -> str:
        """Extract description from template file or generate one."""
        # Check if there's a description in the config
        if 'DESCRIPTION' in config:
            return config['DESCRIPTION']
        
        # Generate description based on filename and contents
        filename = template_path.stem
        if 'template' in filename.lower():
            return "Base template for new projects"
        elif 'bow' in filename.lower():
            return "Bow River watershed configuration"
        elif 'example' in filename.lower():
            return "Example configuration"
        else:
            domain = config.get('DOMAIN_NAME', 'custom')
            return f"Configuration for {domain} domain"

    def update_config(self, config_file: str, updates: Dict[str, Any] = None) -> None:
        """
        Update an existing configuration file with new settings.
        
        Args:
            config_file: Path to configuration file to update
            updates: Dictionary of updates to apply
        """
        import yaml
        
        config_path = Path(config_file)
        
        print(f"\n🔧 Updating Configuration: {config_path}")
        print("=" * 50)
        
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return
        
        try:
            # Load existing config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"📄 Loaded configuration: {config_path}")
            print(f"🏞️  Current domain: {config.get('DOMAIN_NAME', 'Unknown')}")
            print(f"🧪 Current experiment: {config.get('EXPERIMENT_ID', 'Unknown')}")
            
            # Apply updates if provided
            if updates:
                for key, value in updates.items():
                    old_value = config.get(key, 'Not set')
                    config[key] = value
                    print(f"🔄 Updated {key}: {old_value} → {value}")
            
            # Interactive updates (if no updates provided)
            if not updates:
                print("\n💡 Interactive update mode. Press Enter to keep current values.")
                
                # Key settings to potentially update
                update_fields = [
                    ('DOMAIN_NAME', 'Domain name'),
                    ('EXPERIMENT_ID', 'Experiment ID'),
                    ('EXPERIMENT_TIME_START', 'Start time'),
                    ('EXPERIMENT_TIME_END', 'End time'),
                    ('HYDROLOGICAL_MODEL', 'Hydrological model'),
                    ('DOMAIN_DEFINITION_METHOD', 'Domain definition method')
                ]
                
                for field, description in update_fields:
                    current = config.get(field, '')
                    prompt = f"  {description} [{current}]: "
                    new_value = input(prompt).strip()
                    if new_value:
                        config[field] = new_value
                        print(f"    ✅ Updated: {new_value}")
            
            # Create backup
            backup_path = config_path.with_suffix('.yaml.backup')
            import shutil
            shutil.copy2(config_path, backup_path)
            print(f"💾 Backup created: {backup_path}")
            
            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"✅ Configuration updated successfully!")
            print(f"📁 Updated file: {config_path}")
            
        except Exception as e:
            print(f"❌ Error updating configuration: {str(e)}")

    def validate_environment(self) -> None:
        """Validate system environment and dependencies."""
        print("\n🔍 Environment Validation:")
        print("=" * 40)
        
        # Check Python version
        import sys
        python_version = sys.version_info
        print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major >= 3 and python_version.minor >= 8:
            print("   ✅ Python version is compatible")
        else:
            print("   ❌ Python 3.8+ required")
        
        # Check required packages
        required_packages = [
            'numpy', 'pandas', 'geopandas', 'rasterio', 'shapely', 
            'yaml', 'pathlib', 'datetime', 'logging'
        ]
        
        print(f"\n📦 Package Dependencies:")
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} (missing)")
                missing_packages.append(package)
        
        # Check CONFLUENCE structure
        print(f"\n📁 CONFLUENCE Structure:")
        
        required_dirs = [
            'utils', 'utils/project', 'utils/data', 'utils/geospatial',
            'utils/models', 'utils/evaluation', 'utils/optimization', 'utils/cli'
        ]
        
        for directory in required_dirs:
            dir_path = Path(directory)
            if dir_path.exists():
                print(f"   ✅ {directory}/")
            else:
                print(f"   ❌ {directory}/ (missing)")
        
        # Check config files
        print(f"\n⚙️  Configuration Files:")
        config_files = ['0_config_files/config_template.yaml']
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                print(f"   ✅ {config_file}")
            else:
                print(f"   ❌ {config_file} (missing)")
        
        # System resources
        print(f"\n💻 System Resources:")
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            print(f"   💾 RAM: {memory.available // (1024**3):.1f} GB available / {memory.total // (1024**3):.1f} GB total")
            print(f"   💿 Disk: {disk.free // (1024**3):.1f} GB available / {disk.total // (1024**3):.1f} GB total")
            
            if memory.available > 4 * (1024**3):  # 4GB
                print("   ✅ Sufficient memory available")
            else:
                print("   ⚠️  Low memory - consider closing other applications")
                
        except ImportError:
            print("   ⚠️  psutil not available - cannot check system resources")


        # Validate external binaries
        binary_results = self.validate_binaries()
                

        print(f"\n📊 Validation Summary:")
        
        if missing_packages:
            print(f"   ❌ Missing packages: {', '.join(missing_packages)}")
            print(f"   💡 Install with: pip install {' '.join(missing_packages)}")
        else:
            print("   ✅ All Python dependencies satisfied")
        
        # binary summary
        valid_tools = len(binary_results['valid_tools'])
        total_tools = len(self.external_tools)
        missing_tools = len(binary_results['missing_tools'])
        
        if missing_tools > 0:
            print(f"   🔧 External tools: {valid_tools}/{total_tools} available")
            print(f"   💡 Install missing tools: python CONFLUENCE.py --get_executables")
        else:
            print(f"   ✅ All external tools available ({valid_tools}/{total_tools})")
        
        print("   🚀 CONFLUENCE is ready to run!")


        # Summary
        print(f"\n📊 Validation Summary:")
        if missing_packages:
            print(f"   ❌ Missing packages: {', '.join(missing_packages)}")
            print(f"   💡 Install with: pip install {' '.join(missing_packages)}")
        else:
            print("   ✅ All dependencies satisfied")
        
        print("   🚀 CONFLUENCE is ready to run!")

    def _setup_parser(self) -> None:
        """Set up the argument parser with all CLI options."""
        self.parser = argparse.ArgumentParser(
            description='CONFLUENCE - Community Optimization Nexus for Leveraging Understanding of Environmental Networks in Computational Exploration',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples_text()
        )
        
        # Configuration options
        config_group = self.parser.add_argument_group('Configuration Options')
        config_group.add_argument(
            '--config', 
            type=str,
            default='./0_config_files/config_template.yaml',
            help='Path to YAML configuration file (default: ./0_config_files/config_active.yaml)'
        )
        config_group.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug output and detailed logging'
        )
        config_group.add_argument(
            '--version',
            action='version',
            version='CONFLUENCE 1.0.0'
        )
        
        # NEW: Config Management
        config_mgmt_group = self.parser.add_argument_group('Configuration Management')
        config_mgmt_group.add_argument(
            '--list_templates',
            action='store_true',
            help='List all available configuration templates'
        )
        config_mgmt_group.add_argument(
            '--update_config',
            type=str,
            metavar='CONFIG_FILE',
            help='Update an existing configuration file with new settings'
        )
        config_mgmt_group.add_argument(
            '--validate_environment',
            action='store_true',
            help='Validate system environment and dependencies'
        )
        #Binary/Executable Management
        binary_mgmt_group = self.parser.add_argument_group('Binary Management')
        binary_mgmt_group.add_argument(
            '--get_executables',
            nargs='*',
            metavar='TOOL_NAME',
            help='Clone and install external tool repositories (summa, mizuroute, fuse, taudem, gistool, datatool). ' +
                'Use without arguments to install all tools, or specify specific tools.'
        )
        binary_mgmt_group.add_argument(
            '--validate_binaries',
            action='store_true',
            help='Validate that external tool binaries exist and are functional'
        )
        binary_mgmt_group.add_argument(
            '--force_install',
            action='store_true',
            help='Force reinstallation of tools even if they already exist'
        )
        
        # Workflow execution options
        workflow_group = self.parser.add_argument_group('Workflow Execution')
        workflow_group.add_argument(
            '--run_workflow',
            action='store_true',
            help='Run the complete CONFLUENCE workflow (default behavior if no individual steps specified)'
        )
        workflow_group.add_argument(
            '--force_rerun',
            action='store_true',
            help='Force rerun of all steps, overwriting existing outputs'
        )
        workflow_group.add_argument(
            '--stop_on_error',
            action='store_true',
            default=True,
            help='Stop workflow execution on first error (default: True)'
        )
        workflow_group.add_argument(
            '--continue_on_error',
            action='store_true',
            help='Continue workflow execution even if errors occur'
        )
        
        # NEW: Workflow Management
        workflow_mgmt_group = self.parser.add_argument_group('Workflow Management')
        workflow_mgmt_group.add_argument(
            '--workflow_status',
            action='store_true',
            help='Show detailed workflow status with step completion and file checks'
        )
        workflow_mgmt_group.add_argument(
            '--resume_from',
            type=str,
            metavar='STEP_NAME',
            help='Resume workflow execution from a specific step'
        )
        workflow_mgmt_group.add_argument(
            '--clean',
            action='store_true',
            help='Clean intermediate files and outputs'
        )
        workflow_mgmt_group.add_argument(
            '--clean_level',
            type=str,
            choices=['intermediate', 'outputs', 'all'],
            default='intermediate',
            help='Level of cleaning: intermediate files only, outputs, or all (default: intermediate)'
        )
        
        # Individual workflow steps (existing)
        steps_group = self.parser.add_argument_group('Individual Workflow Steps')
        for step_name, step_info in self.workflow_steps.items():
            steps_group.add_argument(
                f'--{step_name}',
                action='store_true',
                help=step_info['description']
            )
        
        # Pour point setup (existing)
        pourpoint_group = self.parser.add_argument_group('Pour Point Setup')
        pourpoint_group.add_argument(
            '--pour_point',
            type=str,
            metavar='LAT/LON',
            help='Set up CONFLUENCE for a pour point coordinate (format: lat/lon, e.g., 51.1722/-115.5717)'
        )
        pourpoint_group.add_argument(
            '--domain_def',
            type=str,
            choices=self.domain_definition_methods,
            help=f'Domain definition method when using --pour_point. Options: {", ".join(self.domain_definition_methods)}'
        )
        pourpoint_group.add_argument(
            '--domain_name',
            type=str,
            help='Domain name when using --pour_point (required)'
        )
        pourpoint_group.add_argument(
            '--experiment_id',
            type=str,
            help='Override experiment ID in configuration'
        )
        pourpoint_group.add_argument(
            '--bounding_box_coords',
            type=str,
            metavar='LAT_MAX/LON_MIN/LAT_MIN/LON_MAX',
            help='Bounding box coordinates (format: lat_max/lon_min/lat_min/lon_max, e.g., 51.76/-116.55/50.95/-115.5). Default: 1 degree buffer around pour point'
        )
        
        # Analysis and status options (updated)
        status_group = self.parser.add_argument_group('Status and Analysis')
        status_group.add_argument(
            '--status',
            action='store_true',
            help='Show current workflow status and exit'
        )
        status_group.add_argument(
            '--list_steps',
            action='store_true',
            help='List all available workflow steps and exit'
        )
        status_group.add_argument(
            '--validate_config',
            action='store_true',
            help='Validate configuration file and exit'
        )
        status_group.add_argument(
            '--dry_run',
            action='store_true',
            help='Show what would be executed without actually running'
        )
        # slurm group 
        slurm_group = self.parser.add_argument_group('SLURM Job Submission')
        slurm_group.add_argument(
            '--submit_job',
            action='store_true',
            help='Submit the execution plan as a SLURM job instead of running locally'
        )
        slurm_group.add_argument(
            '--job_name',
            type=str,
            help='SLURM job name (default: auto-generated from domain and steps)'
        )
        slurm_group.add_argument(
            '--job_time',
            type=str,
            default='48:00:00',
            help='SLURM job time limit (default: 48:00:00)'
        )
        slurm_group.add_argument(
            '--job_nodes',
            type=int,
            default=1,
            help='Number of nodes for SLURM job (default: 1)'
        )
        slurm_group.add_argument(
            '--job_ntasks',
            type=int,
            default=1,
            help='Number of tasks for SLURM job (default: 1 for workflow, auto for calibration)'
        )
        slurm_group.add_argument(
            '--job_memory',
            type=str,
            default='50G',
            help='Memory requirement for SLURM job (default: 50G)'
        )
        slurm_group.add_argument(
            '--job_account',
            type=str,
            help='SLURM account to charge job to (required for most systems)'
        )
        slurm_group.add_argument(
            '--job_partition',
            type=str,
            help='SLURM partition/queue to submit to'
        )
        slurm_group.add_argument(
            '--job_modules',
            type=str,
            default='confluence_modules',
            help='Module to restore in SLURM job (default: confluence_modules)'
        )
        slurm_group.add_argument(
            '--conda_env',
            type=str,
            default='confluence',
            help='Conda environment to activate (default: confluence)'
        )
        slurm_group.add_argument(
            '--submit_and_wait',
            action='store_true',
            help='Submit job and wait for completion (monitors job status)'
        )
        slurm_group.add_argument(
            '--slurm_template',
            type=str,
            help='Custom SLURM script template file to use'
        )
            
    def _get_examples_text(self) -> str:
        """Generate examples text for help output including new binary management commands."""
        return """
Examples:
# Basic workflow execution
python CONFLUENCE.py
python CONFLUENCE.py --config /path/to/config.yaml

# Individual workflow steps
python CONFLUENCE.py --calibrate_model
python CONFLUENCE.py --setup_project --create_pour_point --define_domain

# Pour point setup
python CONFLUENCE.py --pour_point 51.1722/-115.5717 --domain_def delineate --domain_name "MyWatershed"
python CONFLUENCE.py --pour_point 51.1722/-115.5717 --domain_def delineate --domain_name "Test" --bounding_box_coords 52.0/-116.0/51.0/-115.0

# Configuration management
python CONFLUENCE.py --list_templates
python CONFLUENCE.py --update_config my_config.yaml
python CONFLUENCE.py --validate_environment

# Binary/executable management 
python CONFLUENCE.py --get_executables
python CONFLUENCE.py --get_executables summa mizuroute
python CONFLUENCE.py --validate_binaries
python CONFLUENCE.py --get_executables --force_install

# Workflow management
python CONFLUENCE.py --workflow_status
python CONFLUENCE.py --resume_from define_domain
python CONFLUENCE.py --clean --clean_level intermediate
python CONFLUENCE.py --clean --clean_level all --dry_run

# Status and validation
python CONFLUENCE.py --status
python CONFLUENCE.py --list_steps
python CONFLUENCE.py --validate_config

# Advanced options
python CONFLUENCE.py --debug --force_rerun
python CONFLUENCE.py --dry_run
python CONFLUENCE.py --continue_on_error

For more information, visit: https://github.com/DarriEy/CONFLUENCE
    """
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Args:
            args: Optional list of arguments to parse (for testing)
            
        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)
    

    def _validate_bounding_box(self, bbox_string: str) -> bool:
        """
        Validate bounding box coordinate string format.
        
        Args:
            bbox_string: Bounding box string in format "lat_max/lon_min/lat_min/lon_max"
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            parts = bbox_string.split('/')
            if len(parts) != 4:
                return False
            
            lat_max, lon_min, lat_min, lon_max = map(float, parts)
            
            # Basic range and logic validation
            if not (-90 <= lat_min <= lat_max <= 90):
                return False
            if not (-180 <= lon_min <= lon_max <= 180):
                return False
            
            return True
        except (ValueError, IndexError):
            return False
    def _validate_coordinates(self, coord_string: str) -> bool:
        """
        Validate coordinate string format.
        
        Args:
            coord_string: Coordinate string in format "lat/lon"
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            parts = coord_string.split('/')
            if len(parts) != 2:
                return False
            
            lat, lon = float(parts[0]), float(parts[1])
            
            # Basic range validation
            if not (-90 <= lat <= 90):
                return False
            if not (-180 <= lon <= 180):
                return False
            
            return True
        except (ValueError, IndexError):
            return False
        
    def get_execution_plan(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Determine what should be executed based on parsed arguments.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            Dictionary describing the execution plan
        """
        plan = {
            'mode': 'workflow',  # 'workflow', 'individual_steps', 'pour_point_setup', 'status_only', 'management'
            'steps': [],
            'config_overrides': {},
            'settings': {
                'force_rerun': args.force_rerun,
                'stop_on_error': args.stop_on_error and not args.continue_on_error,
                'debug': args.debug,
                'dry_run': args.dry_run
            }
        }
        # Handle binary management operations
        if (hasattr(args, 'get_executables') and args.get_executables is not None) or getattr(args, 'validate_binaries', False):
            plan['mode'] = 'binary_management'
            plan['binary_operations'] = {
                'get_executables': getattr(args, 'get_executables', None),
                'validate_binaries': getattr(args, 'validate_binaries', False),
                'force_install': getattr(args, 'force_install', False)
            }
            return plan
        
        # Handle management operations (config and workflow management)
        if (args.list_templates or args.update_config or args.validate_environment or 
            args.workflow_status or args.resume_from or args.clean):
            plan['mode'] = 'management'
            plan['management_operations'] = {
                'list_templates': args.list_templates,
                'update_config': args.update_config,
                'validate_environment': args.validate_environment,
                'workflow_status': args.workflow_status,
                'resume_from': args.resume_from,
                'clean': args.clean,
                'clean_level': getattr(args, 'clean_level', 'intermediate')
            }
            return plan
        
        # Handle status-only operations
        if args.status or args.list_steps or args.validate_config:
            plan['mode'] = 'status_only'
            plan['status_operations'] = {
                'show_status': args.status,
                'list_steps': args.list_steps,
                'validate_config': args.validate_config
            }
            return plan
        
        # Handle pour point setup
        if args.pour_point:
            plan['mode'] = 'pour_point_setup'
            plan['pour_point'] = {
                'coordinates': args.pour_point,
                'domain_definition_method': args.domain_def,
                'domain_name': args.domain_name,
                'bounding_box_coords': getattr(args, 'bounding_box_coords', None)
            }
        
        
        # Check for individual step execution
        individual_steps = []
        for step_name in self.workflow_steps.keys():
            if getattr(args, step_name, False):
                individual_steps.append(step_name)
        
        if individual_steps:
            plan['mode'] = 'individual_steps'
            plan['steps'] = individual_steps
        elif not args.pour_point and not args.run_workflow:
            # Default to full workflow if no specific steps requested
            plan['mode'] = 'workflow'
        
        # Handle configuration overrides
        if args.domain_name:
            plan['config_overrides']['DOMAIN_NAME'] = args.domain_name
        
        if args.experiment_id:
            plan['config_overrides']['EXPERIMENT_ID'] = args.experiment_id
        
        if args.pour_point:
            plan['config_overrides']['POUR_POINT_COORDS'] = args.pour_point
            plan['config_overrides']['DOMAIN_DEFINITION_METHOD'] = args.domain_def
            
            # Add bounding box to overrides if provided
            if hasattr(args, 'bounding_box_coords') and args.bounding_box_coords:
                plan['config_overrides']['BOUNDING_BOX_COORDS'] = args.bounding_box_coords
        
        if args.force_rerun:
            plan['config_overrides']['FORCE_RUN_ALL_STEPS'] = True
        
            # Handle SLURM job submission
        if args.submit_job:
            plan['mode'] = 'slurm_job'
            plan['slurm_options'] = {
                'job_name': args.job_name,
                'job_time': args.job_time,
                'job_nodes': args.job_nodes,
                'job_ntasks': args.job_ntasks,
                'job_memory': args.job_memory,
                'job_account': args.job_account,
                'job_partition': args.job_partition,
                'job_modules': args.job_modules,
                'conda_env': args.conda_env,
                'submit_and_wait': args.submit_and_wait,
                'slurm_template': args.slurm_template
            }
            # Preserve the original execution plan for the job
            if individual_steps:
                plan['job_mode'] = 'individual_steps'
                plan['job_steps'] = individual_steps
            elif args.pour_point:
                plan['job_mode'] = 'pour_point_setup'
            else:
                plan['job_mode'] = 'workflow'
        
        return plan

    def submit_slurm_job(self, execution_plan: Dict[str, Any], confluence_instance=None) -> Dict[str, Any]:
        """
        Submit a SLURM job based on the execution plan.
        
        Args:
            execution_plan: Execution plan from CLI manager
            confluence_instance: CONFLUENCE instance (optional)
            
        Returns:
            Dictionary with job submission results
        """
        import subprocess
        import tempfile
        from datetime import datetime
        
        print("\n🚀 Submitting CONFLUENCE SLURM Job")
        print("=" * 50)
        
        slurm_options = execution_plan.get('slurm_options', {})
        job_mode = execution_plan.get('job_mode', 'workflow')
        job_steps = execution_plan.get('job_steps', [])
        config_file = execution_plan.get('config_file', './0_config_files/config_template.yaml')
        
        # Generate job name if not provided
        if not slurm_options.get('job_name'):
            domain_name = "CONFLUENCE"
            if confluence_instance and hasattr(confluence_instance, 'config'):
                domain_name = confluence_instance.config.get('DOMAIN_NAME', 'CONFLUENCE')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            slurm_options['job_name'] = f"{domain_name}_{job_mode}_{timestamp}"
        
        # Auto-adjust resources based on job type
        if 'calibrate_model' in job_steps or job_mode == 'workflow':
            # Calibration jobs need more resources
            if slurm_options.get('job_ntasks') == 1:  # If not explicitly set
                slurm_options['job_ntasks'] = min(100, slurm_options.get('job_ntasks', 50))
            if slurm_options.get('job_memory') == '50G':  # If using default
                slurm_options['job_memory'] = '100G'
            if slurm_options.get('job_time') == '48:00:00':  # If using default
                slurm_options['job_time'] = '72:00:00'
        
        # Create SLURM script
        script_content = self._create_confluence_slurm_script(
            execution_plan, slurm_options, config_file
        )
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as script_file:
            script_file.write(script_content)
            script_path = script_file.name
        
        # Make script executable
        import os
        os.chmod(script_path, 0o755)
        
        print(f"📝 Created SLURM script: {script_path}")
        print(f"💼 Job name: {slurm_options['job_name']}")
        print(f"⏰ Time limit: {slurm_options['job_time']}")
        print(f"🖥️  Resources: {slurm_options['job_ntasks']} tasks, {slurm_options['job_memory']} memory")
        
        if execution_plan.get('settings', {}).get('dry_run', False):
            print("\n🔍 DRY RUN - SLURM script content:")
            print("-" * 40)
            print(script_content)
            print("-" * 40)
            return {
                'dry_run': True,
                'script_path': script_path,
                'script_content': script_content,
                'job_name': slurm_options['job_name']
            }
        
        # Check if sbatch is available
        if not self._check_slurm_available():
            raise RuntimeError("SLURM 'sbatch' command not found. Is SLURM installed?")
        
        # Submit job
        try:
            cmd = ['sbatch', script_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Extract job ID
            job_id = None
            for line in result.stdout.split('\n'):
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1].strip()
                    break
            
            if not job_id:
                raise RuntimeError("Could not extract job ID from sbatch output")
            
            print(f"✅ Job submitted successfully!")
            print(f"🆔 Job ID: {job_id}")
            print(f"📋 Check status: squeue -j {job_id}")
            print(f"📊 Monitor logs: tail -f CONFLUENCE_*_{job_id}.{{'out','err'}}")
            
            submission_result = {
                'success': True,
                'job_id': job_id,
                'script_path': script_path,
                'job_name': slurm_options['job_name'],
                'submission_time': datetime.now().isoformat()
            }
            
            # Wait for job completion if requested
            if slurm_options.get('submit_and_wait', False):
                print(f"\n⏳ Waiting for job {job_id} to complete...")
                self._monitor_slurm_job(job_id)
            
            return submission_result
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error submitting job: {e}")
            print(f"stderr: {e.stderr}")
            raise RuntimeError(f"Failed to submit SLURM job: {e}")
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise

    def _create_confluence_slurm_script(self, execution_plan: Dict[str, Any], 
                                    slurm_options: Dict[str, Any], 
                                    config_file: str) -> str:
        """Create SLURM script content for CONFLUENCE workflow."""
        
        job_mode = execution_plan.get('job_mode', 'workflow')
        job_steps = execution_plan.get('job_steps', [])
        config_overrides = execution_plan.get('config_overrides', {})
        
        # Build CONFLUENCE command
        if job_mode == 'individual_steps':
            # Individual steps
            confluence_cmd = f"python CONFLUENCE.py --config {config_file}"
            for step in job_steps:
                confluence_cmd += f" --{step}"
        elif job_mode == 'pour_point_setup':
            # Pour point setup
            pour_point_info = execution_plan.get('pour_point', {})
            confluence_cmd = (
                f"python CONFLUENCE.py "
                f"--pour_point {pour_point_info.get('coordinates')} "
                f"--domain_def {pour_point_info.get('domain_definition_method')} "
                f"--domain_name '{pour_point_info.get('domain_name')}'"
            )
            if pour_point_info.get('bounding_box_coords'):
                confluence_cmd += f" --bounding_box_coords {pour_point_info['bounding_box_coords']}"
        else:
            # Full workflow
            confluence_cmd = f"python CONFLUENCE.py --config {config_file}"
        
        # Add common options
        settings = execution_plan.get('settings', {})
        if settings.get('force_rerun', False):
            confluence_cmd += " --force_rerun"
        if settings.get('debug', False):
            confluence_cmd += " --debug"
        if settings.get('continue_on_error', False):
            confluence_cmd += " --continue_on_error"
        
        # Create SLURM script
        script = f"""#!/bin/bash
    #SBATCH --job-name={slurm_options['job_name']}
    #SBATCH --output=CONFLUENCE_{slurm_options['job_name']}_%j.out
    #SBATCH --error=CONFLUENCE_{slurm_options['job_name']}_%j.err
    #SBATCH --time={slurm_options['job_time']}
    #SBATCH --ntasks={slurm_options['job_ntasks']}
    #SBATCH --mem={slurm_options['job_memory']}"""

        # Add optional SLURM directives
        if slurm_options.get('job_account'):
            script += f"\n#SBATCH --account={slurm_options['job_account']}"
        if slurm_options.get('job_partition'):
            script += f"\n#SBATCH --partition={slurm_options['job_partition']}"
        if slurm_options.get('job_nodes') and slurm_options['job_nodes'] > 1:
            script += f"\n#SBATCH --nodes={slurm_options['job_nodes']}"

        script += f"""

    # Job information
    echo "=========================================="
    echo "CONFLUENCE SLURM Job Started"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Job Name: {slurm_options['job_name']}"
    echo "Node: $HOSTNAME"
    echo "Started: $(date)"
    echo "Working Directory: $(pwd)"
    echo "=========================================="

    # Load modules and environment
    echo "Loading environment..."
    """
        
        # Add module loading
        if slurm_options.get('job_modules'):
            script += f"module restore {slurm_options['job_modules']}\n"
        
        # Add conda environment activation
        if slurm_options.get('conda_env'):
            script += f"conda activate {slurm_options['conda_env']}\n"
        
        script += f"""
    echo "Python environment: $(which python)"
    echo "CONFLUENCE directory: $(pwd)"
    echo ""

    # Run CONFLUENCE workflow
    echo "Starting CONFLUENCE workflow..."
    echo "Command: {confluence_cmd}"
    echo ""

    {confluence_cmd}

    exit_code=$?

    echo ""
    echo "=========================================="
    echo "CONFLUENCE Job Completed"
    echo "Exit code: $exit_code"
    echo "Finished: $(date)"
    echo "=========================================="

    exit $exit_code
    """
    
        return script

    def _check_slurm_available(self) -> bool:
        """Check if SLURM commands are available."""
        import shutil
        return shutil.which('sbatch') is not None

    def _monitor_slurm_job(self, job_id: str) -> None:
        """Monitor SLURM job until completion."""
        import subprocess
        import time
        
        print(f"🔄 Monitoring job {job_id}...")
        
        while True:
            try:
                # Check if job is still in queue
                result = subprocess.run(
                    ['squeue', '-j', job_id, '-h'], 
                    capture_output=True, text=True
                )
                
                if not result.stdout.strip():
                    # Job no longer in queue
                    print(f"✅ Job {job_id} completed!")
                    
                    # Check final status
                    try:
                        status_result = subprocess.run(
                            ['sacct', '-j', job_id, '-o', 'State', '-n'],
                            capture_output=True, text=True
                        )
                        final_status = status_result.stdout.strip().split('\n')[0]
                        print(f"📊 Final status: {final_status}")
                        
                        if 'COMPLETED' not in final_status:
                            print(f"⚠️ Job may have failed. Check logs for details.")
                    
                    except subprocess.SubprocessError:
                        pass
                    
                    break
                else:
                    # Parse queue status
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        status_info = lines[0].split()
                        if len(status_info) >= 5:
                            status = status_info[4]  # Job state
                            print(f"⏳ Job status: {status}")
                    
            except subprocess.SubprocessError as e:
                print(f"⚠️ Error checking job status: {e}")
            
            # Wait before next check
            time.sleep(60)  # Check every minute

    def handle_slurm_job_submission(self, execution_plan: Dict[str, Any], 
                                confluence_instance=None) -> bool:
        """
        Handle SLURM job submission workflow.
        
        Args:
            execution_plan: Execution plan from CLI manager
            confluence_instance: CONFLUENCE instance (optional)
            
        Returns:
            bool: Success status
        """
        try:
            # Validate SLURM options
            slurm_options = execution_plan.get('slurm_options', {})
            
            # Check for required options on some systems
            if not slurm_options.get('job_account'):
                print("⚠️ Warning: No SLURM account specified. This may be required on your system.")
                print("   Use --job_account to specify an account if job submission fails.")
            
            # Submit the job
            result = self.submit_slurm_job(execution_plan, confluence_instance)
            
            if result.get('success', False):
                print(f"\n🎉 SLURM job submission successful!")
                if not slurm_options.get('submit_and_wait', False):
                    print(f"💡 Job is running in background. Monitor with:")
                    print(f"   squeue -j {result['job_id']}")
                    print(f"   tail -f CONFLUENCE_*_{result['job_id']}.out")
                return True
            else:
                print(f"❌ Job submission failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in SLURM job submission: {str(e)}")
            return False

    def validate_arguments(self, args: argparse.Namespace) -> Tuple[bool, List[str]]:
        """
        Validate parsed arguments for logical consistency.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check pour point format
        if args.pour_point:
            if not self._validate_coordinates(args.pour_point):
                errors.append(f"Invalid pour point format: {args.pour_point}. Expected format: lat/lon (e.g., 51.1722/-115.5717)")
            
            if not args.domain_def:
                errors.append("--domain_def is required when using --pour_point")
                
            if not args.domain_name:
                errors.append("--domain_name is required when using --pour_point")
            
            # Validate bounding box if provided
            if hasattr(args, 'bounding_box_coords') and args.bounding_box_coords and not self._validate_bounding_box(args.bounding_box_coords):
                errors.append(f"Invalid bounding box format: {args.bounding_box_coords}. Expected format: lat_max/lon_min/lat_min/lon_max")
        
        # Validate resume_from step name
        if args.resume_from:
            if args.resume_from not in self.workflow_steps:
                errors.append(f"Invalid step name for --resume_from: {args.resume_from}. Available steps: {', '.join(self.workflow_steps.keys())}")
        
        # Validate update_config file exists
        if args.update_config:
            config_path = Path(args.update_config)
            if not config_path.exists():
                errors.append(f"Configuration file not found for --update_config: {config_path}")
        
        # Check conflicting options
        if args.stop_on_error and args.continue_on_error:
            errors.append("Cannot specify both --stop_on_error and --continue_on_error")
        
        # Check if operations that don't need config files are being run
        binary_management_ops = (
            (hasattr(args, 'get_executables') and args.get_executables is not None) or 
            getattr(args, 'validate_binaries', False)
        )
        
        standalone_management_ops = (
            args.list_templates or 
            args.validate_environment or
            args.update_config
        )
        
        status_only_ops = (
            args.list_steps or 
            (args.validate_config and not args.pour_point)
        )
        
        # Only validate config file if we actually need it
        needs_config_file = not (binary_management_ops or standalone_management_ops or status_only_ops)
        
        if needs_config_file and not args.pour_point:
            config_path = Path(args.config)
            if not config_path.exists():
                errors.append(f"Configuration file not found: {config_path}")
        
        return len(errors) == 0, errors
    
    def apply_config_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configuration overrides from CLI arguments.
        
        Args:
            config: Original configuration dictionary
            overrides: Override values from CLI
            
        Returns:
            Updated configuration dictionary
        """
        updated_config = config.copy()
        updated_config.update(overrides)
        return updated_config
    
    def print_status_information(self, confluence_instance, operations: Dict[str, bool]) -> None:
        """
        Print various status information based on requested operations.
        
        Args:
            confluence_instance: CONFLUENCE system instance
            operations: Dictionary of status operations to perform
        """
        if operations.get('list_steps'):
            self._print_workflow_steps()
        
        if operations.get('validate_config'):
            self._print_config_validation(confluence_instance)
        
        if operations.get('show_status'):
            self._print_workflow_status(confluence_instance)
    
    def _print_workflow_steps(self) -> None:
        """Print all available workflow steps."""
        print("\n📋 Available Workflow Steps:")
        print("=" * 50)
        for step_name, step_info in self.workflow_steps.items():
            print(f"--{step_name:<25} {step_info['description']}")
        print("\n💡 Use these flags to run individual steps, e.g.:")
        print("  python CONFLUENCE.py --setup_project --create_pour_point")
        print()
    
    def _print_config_validation(self, confluence_instance) -> None:
        """Print configuration validation results."""
        print("\n🔍 Configuration Validation:")
        print("=" * 30)
        
        # This would integrate with the CONFLUENCE validation methods
        if hasattr(confluence_instance, 'workflow_orchestrator'):
            is_valid = confluence_instance.workflow_orchestrator.validate_workflow_prerequisites()
            if is_valid:
                print("✅ Configuration is valid")
            else:
                print("❌ Configuration validation failed")
                print("Check logs for detailed error information")
        else:
            print("⚠️  Configuration validation not available")
    
    def _print_workflow_status(self, confluence_instance) -> None:
        """Print current workflow status."""
        print("\n📊 Workflow Status:")
        print("=" * 20)
        
        if hasattr(confluence_instance, 'get_status'):
            status = confluence_instance.get_status()
            print(f"🏞️  Domain: {status.get('domain', 'Unknown')}")
            print(f"🧪 Experiment: {status.get('experiment', 'Unknown')}")
            print(f"⚙️  Config Valid: {'✅' if status.get('config_valid', False) else '❌'}")
            print(f"🔧 Managers Initialized: {'✅' if status.get('managers_initialized', False) else '❌'}")
            print(f"📁 Config Path: {status.get('config_path', 'Unknown')}")
            print(f"🐛 Debug Mode: {'✅' if status.get('debug_mode', False) else '❌'}")
            
            if 'workflow_status' in status:
                workflow_status = status['workflow_status']
                print(f"🔄 Workflow Status: {workflow_status}")
        else:
            print("⚠️  Status information not available")
        
    def setup_pour_point_workflow(self, coordinates: str, domain_def_method: str, domain_name: str, 
                                bounding_box_coords: Optional[str] = None, 
                                confluence_code_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a configuration setup for pour point workflow.
        
        This method:
        1. Loads the config template from CONFLUENCE_CODE_DIR/0_config_files/config_template.yaml
        2. Updates key settings (pour point, domain name, domain definition method, bounding box)
        3. Saves as config_{domain_name}.yaml
        4. Returns configuration details for workflow execution
        
        Args:
            coordinates: Pour point coordinates in "lat/lon" format
            domain_def_method: Domain definition method to use
            domain_name: Name for the domain/watershed
            bounding_box_coords: Optional bounding box, defaults to 1-degree buffer around pour point
            confluence_code_dir: Path to CONFLUENCE code directory
            
        Returns:
            Dictionary with pour point workflow configuration including 'config_file' path
        """
        import yaml
        from pathlib import Path
        
        try:
            print(f"\n🎯 Setting up pour point workflow:")
            print(f"   📍 Coordinates: {coordinates}")
            print(f"   🗺️  Domain Definition Method: {domain_def_method}")
            print(f"   🏞️  Domain Name: {domain_name}")
            
            # Parse coordinates for bounding box calculation
            lat, lon = map(float, coordinates.split('/'))
            
            # Calculate bounding box if not provided (1-degree buffer)
            if not bounding_box_coords:
                lat_max = lat + 0.5
                lat_min = lat - 0.5
                lon_max = lon + 0.5
                lon_min = lon - 0.5
                bounding_box_coords = f"{lat_max}/{lon_min}/{lat_min}/{lon_max}"
                print(f"   📦 Auto-calculated bounding box (1° buffer): {bounding_box_coords}")
            else:
                print(f"   📦 User-provided bounding box: {bounding_box_coords}")
            
            # Determine template path
            template_path = None
            
            # Try multiple locations for the template
            possible_locations = [
                Path("./0_config_files/config_template.yaml"),
                Path("../0_config_files/config_template.yaml"), 
                Path("../../0_config_files/config_template.yaml"),
            ]
            
            # If confluence_code_dir provided, try that first
            if confluence_code_dir:
                possible_locations.insert(0, Path(confluence_code_dir) / "0_config_files" / "config_template.yaml")
            
            # Try to detect CONFLUENCE_CODE_DIR from environment or relative paths
            for location in possible_locations:
                if location.exists():
                    template_path = location
                    break
            
            if not template_path:
                raise FileNotFoundError(
                    f"Config template not found. Tried locations: {[str(p) for p in possible_locations]}\n"
                    f"Please ensure you're running from the CONFLUENCE directory or specify --config with a template path."
                )
            
            print(f"   📄 Loading template from: {template_path}")
            
            # Load template config
            with open(template_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Ensure paths are valid before proceeding
            config = self._ensure_valid_config_paths(config, template_path)
            
            # Update config with pour point settings
            config['DOMAIN_NAME'] = domain_name
            config['POUR_POINT_COORDS'] = coordinates
            config['DOMAIN_DEFINITION_METHOD'] = domain_def_method
            config['BOUNDING_BOX_COORDS'] = bounding_box_coords
            
            # Set some sensible defaults for pour point workflows
            if 'EXPERIMENT_ID' not in config or config['EXPERIMENT_ID'] == 'run_1':
                config['EXPERIMENT_ID'] = 'pour_point_setup'
            
            # Save new config file
            output_config_path = Path(f"0_config_files/config_{domain_name}.yaml")
            with open(output_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"   💾 Created config file: {output_config_path}")
            print(f"   ✅ Pour point workflow setup complete!")
            print(f"\n🚀 Next steps:")
            print(f"   1. Review the generated config file: {output_config_path}")
            print(f"   2. CONFLUENCE will now use this config to run the pour point workflow")
            print(f"   3. Essential steps (setup_project, create_pour_point, define_domain, discretize_domain) will be executed")
            
            return {
                'config_file': str(output_config_path.resolve()),  # Return absolute path
                'coordinates': coordinates,
                'domain_name': domain_name,
                'domain_definition_method': domain_def_method,
                'bounding_box_coords': bounding_box_coords,
                'template_used': str(template_path),
                'setup_steps': [
                    'setup_project',
                    'create_pour_point', 
                    'define_domain',
                    'discretize_domain'
                ]
            }
            
        except Exception as e:
            print(f"❌ Error setting up pour point workflow: {str(e)}")
            raise

def create_cli_manager() -> CLIArgumentManager:
    """
    Factory function to create a CLI argument manager instance.
    
    Returns:
        Configured CLIArgumentManager instance
    """
    return CLIArgumentManager()


# Example usage and testing
if __name__ == "__main__":
    # This allows the utility to be tested independently
    cli_manager = CLIArgumentManager()
    
    # Example: parse some test arguments
    test_args = ['--calibrate_model', '--debug']
    args = cli_manager.parse_arguments(test_args)
    
    # Get execution plan
    plan = cli_manager.get_execution_plan(args)
    
    print("Test execution plan:")
    print(f"Mode: {plan['mode']}")
    print(f"Steps: {plan['steps']}")
    print(f"Settings: {plan['settings']}")