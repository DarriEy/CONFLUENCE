# INSTALLATION & ENVIRONMENT SETUP (INSTALL.md)

> **Disclaimer**  
> CONFLUENCE will *attempt* to install and wire up external tooling automatically, but HPC and local environments vary widely. The dev team cannot guarantee successful installation in every environment. Use the recipes below as a starting point and adapt to your system.

---

## 1) Prerequisites

**Build toolchain**
- C/C++ compiler (GCC/Clang)
- Fortran compiler (gfortran)
- CMake ≥ 3.20
- Make/Ninja
- MPI implementation (OpenMPI or MPICH)

**Core libraries / tools**
- GDAL (headers + libs)
- HDF5, NetCDF (C + Fortran)
- BLAS/LAPACK (OpenBLAS or Netlib)
- Python 3.11; **pip** (the installer will create & manage a virtualenv automatically)
- R (optional, for R-based workflows)
- CDO (optional, for climate operators)

**Environment variables you may set** (optional but sometimes helpful)
```bash

# If you need to force compilers (esp. on macOS/Homebrew)
export CC=gcc
export CXX=g++
export FC=gfortran

# MPI wrappers (only when your toolchain needs them)
export MPICC=mpicc
export MPICXX=mpicxx
export MPIFC=mpif90
```

---

## 2) How the installer handles Python

- Running `./confluence --install` will automatically:
  - create a `.venv/` (if none exists),
  - install Python dependencies with **pip**, and
  - reuse the existing `.venv/` on subsequent runs.
- Conda is **not required**. If you prefer Conda, you may still use it, but **pip+venv is the recommended and supported path**.

---

## 3) HPC Module Recipes

> These are known‑good examples. Versions may change on your cluster; adjust as needed.

### A) **Anvil** (Purdue RCAC)
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

### B) **ARC (University of Calgary)**
```bash
# Initialize the site module system
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

### C) **FIR (Compute Canada)**
```bash
# Standard environment
module load StdEnv/2023

# Primary apps and libraries
module load python/3.11.5
module load gdal/3.9.1
module load r/4.5.0
module load cdo/2.2.2
module load mpi4py/4.0.3
module load netcdf-fortran/4.6.1
module load openblas/0.3.24
```

> After loading the modules, run the installer:
```bash
./confluence --install
```

---

## 4) macOS (Apple Silicon or Intel)

### A) Toolchain & package manager
```bash
# Xcode command line tools
xcode-select --install

# Homebrew (see brew.sh for the latest)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### B) System packages via Homebrew
```bash
brew update
brew install cmake gcc open-mpi gdal hdf5 netcdf netcdf-fortran openblas lapack cdo
brew install r
```

> Tip: On Apple Silicon, `gcc-14`, `g++-14`, `gfortran-14` are typical. Adjust if Homebrew ships newer.

```bash
# (Optional) pin compilers if CMake picks up Clang when you need GCC
export CC=$(brew --prefix)/bin/gcc-14
export CXX=$(brew --prefix)/bin/g++-14
export FC=$(brew --prefix)/bin/gfortran-14
```

### C) Python environment (no manual steps required)
- Just run the installer — it will create `.venv/` and install with **pip** automatically.

```bash
./confluence --install
```

> If you must manage the venv manually: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

---

## Support stance
- We aim for a one‑command setup, but **we cannot guarantee** success on every system. 