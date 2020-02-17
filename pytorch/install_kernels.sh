#/usr/bin/sh
cd ./unoptimized/kernels/cuda
python setup.py install
cd -

cd ./deepshift/kernels/cpu
python setup.py install
cd -

cd ./deepshift/kernels/cuda
python setup.py install
cd -
