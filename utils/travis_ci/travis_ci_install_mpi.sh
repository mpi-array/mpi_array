#!/usr/bin/env bash

set -e
export MPI_IMPL_AND_VERSION=${1}
export MPI_IMPL=${MPI_IMPL_AND_VERSION%-*} # retain the part before the last dash
export MPI_IMPL_VERSION=${MPI_IMPL_AND_VERSION##*-} # retain the part after the last dash
export MPI_IMPL_MAJ_DOT_MIN=${MPI_IMPL_VERSION%.*} # retain the part before the last dot

export INSTALL_PREFIX=${2}
export BUILD_PREFIX=${3}
export MPI_INSTALL_PREFIX=${INSTALL_PREFIX}/${MPI_IMPL_AND_VERSION}

echo MPI_IMPL_AND_VERSION = ${MPI_IMPL_AND_VERSION}
echo MPI_IMPL             = ${MPI_IMPL}
echo MPI_IMPL_VERSION     = ${MPI_IMPL_VERSION}
echo MPI_IMPL_MAJ_DOT_MIN = ${MPI_IMPL_MAJ_DOT_MIN}
echo INSTALL_PREFIX       = ${INSTALL_PREFIX}
echo MPI_INSTALL_PREFIX   = ${MPI_INSTALL_PREFIX}
echo BUILD_PREFIX         = ${BUILD_PREFIX}

cd ${BUILD_PREFIX}

export MPI_PREFIX=${INSTALL_PREFIX}/${MPI}
export PATH=${MPI_PREFIX}/bin:$PATH
export LIBRARY_PATH=${MPI_PREFIX}/lib:${MPI_PREFIX}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${MPI_PREFIX}/lib:${MPI_PREFIX}/lib64:$LD_LIBRARY_PATH
export CPATH=${MPI_PREFIX}/include:$CPATH

case `uname` in
Linux)
  case ${MPI_IMPL} in
    mpich2) set -x;
      if [ ! -f "${MPI_INSTALL_PREFIX}/bin/mpiexec" ] || ! "${MPI_INSTALL_PREFIX}/bin/mpiexec" "--version" ; then
        echo "Building mpich2 version ${MPI_IMPL_VERSION}..."
        rm -rf ${MPI_INSTALL_PREFIX};
        wget http://www.mpich.org/static/downloads/${MPI_IMPL_VERSION}/mpich2-${MPI_IMPL_VERSION}.tar.gz && \
        tar -xzf mpich2-${MPI_IMPL_VERSION}.tar.gz && \
        cd mpich2-${MPI_IMPL_VERSION} && \
        ./configure \
            --enable-shared \
            --disable-static \
            --disable-f77 \
            --disable-fc \
            --quiet \
            --enable-silent-rules \
            --prefix=${MPI_INSTALL_PREFIX} && \
        make V=0 && make install
      fi;

      ;;
    mpich) set -x;
      if [ ! -f "${MPI_INSTALL_PREFIX}/bin/mpiexec" ] || ! "${MPI_INSTALL_PREFIX}/bin/mpiexec" "--version" ; then
        echo "Building mpich version ${MPI_IMPL_VERSION}..."
        rm -rf ${MPI_INSTALL_PREFIX};
        wget http://www.mpich.org/static/downloads/${MPI_IMPL_VERSION}/mpich-${MPI_IMPL_VERSION}.tar.gz && \
        tar -xzf mpich-${MPI_IMPL_VERSION}.tar.gz && \
        cd mpich-${MPI_IMPL_VERSION} && \
        ./configure \
            --enable-shared \
            --disable-static \
            --disable-fortran \
            --quiet \
            --enable-silent-rules \
            --prefix=${MPI_INSTALL_PREFIX} && \
        make V=0 && make install
      fi;

      ;;
    openmpi) set -x;
      if [ ! -f "${MPI_INSTALL_PREFIX}/bin/ompi_info" ] || ! "${MPI_INSTALL_PREFIX}/bin/ompi_info"; then
        rm -rf ${MPI_INSTALL_PREFIX};
        wget https://www.open-mpi.org/software/ompi/v${MPI_IMPL_MAJ_DOT_MIN}/downloads/openmpi-${MPI_IMPL_VERSION}.tar.bz2 && \
        tar -xjf openmpi-${MPI_IMPL_VERSION}.tar.bz2 && \
        cd openmpi-${MPI_IMPL_VERSION} && \
        ./configure --quiet --enable-silent-rules --prefix=${MPI_INSTALL_PREFIX} && \
        make V=0 && make install
      fi;
      ;;
    *)
      echo "Unknown MPI implementation:" ${MPI_IMPL} \(${MPI_IMPL_AND_VERSION}\)
      exit 1
      ;;
  esac
  ;;
esac
