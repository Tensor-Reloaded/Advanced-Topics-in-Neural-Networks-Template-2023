#!/usr/bin/env bash

export ONEAPI_ROOT=/opt/intel/oneapi
export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest
export MKLROOT=${ONEAPI_ROOT}/mkl/latest
export IPEX_XPU_ONEDNN_LAYOUT=1
source ${ONEAPI_ROOT}/setvars.sh > /dev/null
source ./venv/bin/activate
