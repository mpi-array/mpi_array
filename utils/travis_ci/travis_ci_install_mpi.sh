#!/usr/bin/env sh
# Borrowed from mpi4py:
# https://github.com/mpi4py/mpi4py/blob/master/conf/ci/travis-ci/install-mpi.sh
#

set -e
case `uname` in
Linux)
  sudo apt-get update -q
  case $1 in
    mpich) set -x;
      sudo apt-get install -y -q mpich libmpich-dev
      ;;
    openmpi) set -x;
      sudo apt-get install -y -q openmpi-bin libopenmpi-dev
      ;;
    *)
      echo "Unknown MPI implementation:" $1
      exit 1
      ;;
  esac
  ;;