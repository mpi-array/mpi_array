#!/usr/bin/env bash

set -e
export MPI_IMPL_AND_VERSION=${1}
export MPI_IMPL=${MPI_IMPL_AND_VERSION%-*} # retain the part before the last dash
export MPI_IMPL_VERSION=${MPI_IMPL_AND_VERSION##*-} # retain the part after the last dash
export MPI_IMPL_MAJ_DOT_MIN=${MPI_IMPL_VERSION%.*} # retain the part before the last dot

export INSTALL_PREFIX=${2}
export BUILD_PREFIX=${3}

echo MPI_IMPL_AND_VERSION = ${MPI_IMPL_AND_VERSION}
echo MPI_IMPL             = ${MPI_IMPL}
echo MPI_IMPL_VERSION     = ${MPI_IMPL_VERSION}
echo MPI_IMPL_MAJ_DOT_MIN = ${MPI_IMPL_MAJ_DOT_MIN}
echo INSTALL_PREFIX       = ${INSTALL_PREFIX}
echo BUILD_PREFIX         = ${BUILD_PREFIX}

cd ${BUILD_PREFIX}

case `uname` in
Linux)
  case ${MPI_IMPL} in
    mpich) set -x;
      echo "Building mpich from source not implemented."
      exit 1
      ;;
    openmpi) set -x;
      if [ ! -d "openmpi-${MPI_IMPL_VERSION}" ]; then
        wget https://www.open-mpi.org/software/ompi/v${MPI_IMPL_MAJ_DOT_MIN}/downloads/openmpi-${MPI_IMPL_VERSION}.tar.bz2;
        tar -xjf openmpi-${MPI_IMPL_VERSION}.tar.bz2;
      fi; 
      cd openmpi-${MPI_IMPL_VERSION} && \
      ./configure --quiet --enable-silent-rules --prefix=${INSTALL_PREFIX} && \
      make V=0 && make install
      ;;
    *)
      echo "Unknown MPI implementation:" ${MPI_IMPL} \(${MPI_IMPL_AND_VERSION}\)
      exit 1
      ;;
  esac
  ;;
esac
