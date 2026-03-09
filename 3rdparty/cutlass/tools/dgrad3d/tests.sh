#!/bin/tcsh
rm -f main.o
$CUDA_PATH/bin/nvcc -o main.o $1 -std=c++11 -gencode=arch=compute_70,code=\"sm_70,compute_70\" -Xptxas -v -O3 -I$CUDNN_PATH/include -L$CUDNN_PATH/lib64 -lcudnn -maxrregcount 255
rm -f README.md
echo "# Dgrad3D" > README.md
echo "## Environment" >> README.md
sudo nvidia-smi -pm 1 >> README.md
sudo nvidia-smi -ac 877,1290 >> README.md
sudo nvidia-smi -lgc 1290,1290 >> README.md
nvidia-smi -q -d CLOCK >> README.md
echo $CUDA_PATH >> README.md
echo $CUDNN_PATH >> README.md
sleep 3s
#./main.o mode N C D H W K T R S paddingFront paddingBack paddingTop paddingBottom paddingLeft paddingRight strideD strideH strideW dilationD dilationH dilationW runs crc_check
echo "## CRC check" >> README.md
./main.o cross 1 32 2 2 2  7 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 1 1 >> README.md
./main.o conv  2 33 2 2 2  8 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 1 1 >> README.md
./main.o cross 3 34 2 2 2  9 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 1 1 >> README.md
./main.o conv  4 35 2 2 2 10 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 1 1 >> README.md
./main.o cross 5 36 6 6 6 11 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 1 1 >> README.md
./main.o conv  6 37 6 6 6 12 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 1 1 >> README.md
if ( $2 == 0 ) then
    echo "### cross-correlation without padding" >> README.md
    ./main.o cross 32  16 64 128 128  32 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32  32 32  64  64  32 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32  32 32  64  64  64 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32  64 16  32  32  64 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32  64 16  32  32 128 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32 128  8  16  16 128 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32 128  8  16  16 256 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32 256  4   8   8 256 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    echo "### convolution without padding" >> README.md
    ./main.o conv  32  16 64 128 128  32 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32  32 32  64  64  32 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32  32 32  64  64  64 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32  64 16  32  32  64 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32  64 16  32  32 128 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32 128  8  16  16 128 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32 128  8  16  16 256 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32 256  4   8   8 256 2 2 2 0 0 0 0 0 0 2 2 2 1 1 1 100 0 >> README.md
    echo "### cross-correlation with padding" >> README.md
    ./main.o cross 32  16 64 128 128  32 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32  32 32  64  64  32 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32  32 32  64  64  64 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32  64 16  32  32  64 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32  64 16  32  32 128 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32 128  8  16  16 128 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32 128  8  16  16 256 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o cross 32 256  4   8   8 256 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    echo "### convolution without padding" >> README.md
    ./main.o conv  32  16 64 128 128  32 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32  32 32  64  64  32 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32  32 32  64  64  64 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32  64 16  32  32  64 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32  64 16  32  32 128 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32 128  8  16  16 128 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32 128  8  16  16 256 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
    ./main.o conv  32 256  4   8   8 256 2 2 2 1 1 1 1 1 1 2 2 2 1 1 1 100 0 >> README.md
endif
rm -f main.o
