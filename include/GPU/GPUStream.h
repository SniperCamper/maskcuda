#ifndef GPUSTREAM_H
#define GPUSTREAM_H

#include <cuda_runtime.h>

struct CUDAStream {
    cudaStream_t stream;
    unsigned char* d_private_key;
    unsigned char* d_bitcoin_address;
    int* d_match_found;
    bool in_use;
};

#endif // GPUSTREAM_H
