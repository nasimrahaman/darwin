#!/bin/bash
# start this script from the root directory of the 

if [ -e darwin/core/kruskals_ws.pyx ]
then
    echo "switching directory to darwin/core"
    cd darwin/core
    echo "start compiling kruskals_ws"
elif [ -e ../darwin/core/kruskals_ws.pyx ]
then
    echo "switching directory to ../darwin/core"
    cd ../darwin/core
else
    echo "ERROR: unable to find  kruskals_ws.pyx"
    echo "Please execute script from root directory of library"
    exit
fi

echo "start compiling kruskals_ws.pyx"
cython -a kruskals_ws.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -lstdc++ -fno-strict-aliasing -I/usr/include/python2.7 -o kruskals_ws.so kruskals_ws.c
rm kruskals_ws.c kruskals_ws.html