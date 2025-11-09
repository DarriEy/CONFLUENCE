# INSTALLATION & ENVIRONMENT SETUP

**Note:** SYMFLUENCE attempts to install and configure external tools automatically, but HPC environments vary. Use these recipes as starting points and adjust for your system.

---

## 1. Prerequisites

**Build toolchain**
- GCC or Clang (C/C++), gfortran
- CMake ≥ 3.20
- MPI (OpenMPI or MPICH)

**Core libraries**
- GDAL (headers + libs)
- HDF5, NetCDF (C + Fortran)
- BLAS/LAPACK
- Python 3.11 (pip required)
- R (optional, for R-based components)
- CDO

**Optional environment variables**
```bash
export CC=gcc
export CXX=g++
export FC=gfortran
export MPICC=mpicc
export MPICXX=mpicxx
export MPIFC=mpif90
```

---

## 2. Python Environment

The installer handles Python setup automatically.

Running:
```bash
./symfluence --install
```
will:
- Create a `.venv/` (if not present)
- Install Python dependencies using pip
- Reuse the environment on subsequent runs

**Conda is not required.**  
You may use Conda manually, but `pip + venv` is the supported path.

---

## 3. HPC Module Recipes

Below are verified module configurations for supported clusters.

### A) Anvil (Purdue RCAC)
```bash
module load r/4.4.1
module load gcc/14.2.0
module load openmpi/4.1.6
module load gdal/3.10.0
module load conda/2024.09
module load openblas/0.3.17
module load netcdf-fortran/4.5.3
module load udunits/2.2.28
```

### B) ARC (University of Calgary)
```bash
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

module load gcc/14.2.0
module load cmake
module load netcdf-fortran/4.6.1
module load netcdf-c/4.9.2
module load openblas/0.3.27
module load hdf5/1.14.3
module load gdal/3.9.2
module load netlib-lapack/3.11.0
module load openmpi/4.1.6
module load python/3.11.7
module load r/4.4.1
```

### C) FIR (Compute Canada)
```bash
module load StdEnv/2023
module load python/3.11.5
module load gdal/3.9.1
module load r/4.5.0
module load cdo/2.2.2
module load mpi4py/4.0.3
module load netcdf-fortran/4.6.1
module load openblas/0.3.24
```

After loading modules:
```bash
./symfluence --install
```

---

## 4. macOS (Intel or Apple Silicon)

### A) Toolchain and Package Manager
```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### B) Core Packages
```bash
brew update
brew install cmake gcc open-mpi gdal hdf5 netcdf netcdf-fortran openblas lapack cdo r
```

### C) Optional Compiler Pinning
```bash
export CC=$(brew --prefix)/bin/gcc-14
export CXX=$(brew --prefix)/bin/g++-14
export FC=$(brew --prefix)/bin/gfortran-14
```

### D) Python Setup
Run:
```bash
./symfluence --install
```

If needed manually:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Support Stance
SYMFLUENCE aims for a one-command setup but cannot guarantee success on all systems.  
If installation fails, use your cluster’s module system and re-run the installer.
