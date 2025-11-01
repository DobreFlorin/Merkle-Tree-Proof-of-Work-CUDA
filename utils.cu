#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 2;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src){
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src){
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
    size_t input_length = d_strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}

// CUDA kernel to compute SHA256 hash for each transaction
__global__ void compute_transaction_hashes_kernel(int transaction_size, BYTE *transactions, BYTE *hashes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        apply_sha256(transactions + idx * transaction_size, hashes + idx * SHA256_HASH_SIZE);
    }
}

// CUDA kernel to combine hashes and compute the next level of the Merkle tree
__global__ void combine_hashes_kernel(BYTE *hashes, BYTE *new_hashes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2 || (n % 2 == 1 && idx == n / 2)) {
        BYTE combined[SHA256_HASH_SIZE * 2];
        if (idx * 2 + 1 < n) {
            // Combine two consecutive hashes
            d_strcpy((char *)combined, (const char *)(hashes + idx * 2 * SHA256_HASH_SIZE));
            d_strcat((char *)combined, (const char *)(hashes + (idx * 2 + 1) * SHA256_HASH_SIZE));
        } else {
            // If odd number of hashes, duplicate the last one
            d_strcpy((char *)combined, (const char *)(hashes + idx * 2 * SHA256_HASH_SIZE));
            d_strcat((char *)combined, (const char *)(hashes + idx * 2 * SHA256_HASH_SIZE));
        }
        apply_sha256(combined, new_hashes + idx * SHA256_HASH_SIZE);
    }
}

void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]) {
    // Allocate device memory
    BYTE *d_transactions, *d_hashes, *d_new_hashes;
    size_t transactions_size = n * transaction_size;
    size_t hashes_size = max_transactions_in_a_block * SHA256_HASH_SIZE;
    
    cudaMalloc((void **)&d_transactions, transactions_size);
    cudaMalloc((void **)&d_hashes, hashes_size);
    cudaMalloc((void **)&d_new_hashes, hashes_size);
    
    // Copy transactions to device
    cudaMemcpy(d_transactions, transactions, transactions_size, cudaMemcpyHostToDevice);
    
    // Compute hash for each transaction
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    compute_transaction_hashes_kernel<<<blocksPerGrid, threadsPerBlock>>>(transaction_size, d_transactions, d_hashes, n);
    cudaDeviceSynchronize();
    
    // Allocate host memory for intermediate results
    BYTE *h_hashes = (BYTE *)malloc(hashes_size);
    if (!h_hashes) {
        fprintf(stderr, "Error: Unable to allocate memory for hashes\n");
        cudaFree(d_transactions);
        cudaFree(d_hashes);
        cudaFree(d_new_hashes);
        exit(EXIT_FAILURE);
    }
    
    // Build the Merkle tree
    int current_n = n;
    while (current_n > 1) {
        blocksPerGrid = (current_n + threadsPerBlock - 1) / threadsPerBlock;
        combine_hashes_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_new_hashes, current_n);
        cudaDeviceSynchronize();
        
        // Swap pointers for next iteration
        BYTE *temp = d_hashes;
        d_hashes = d_new_hashes;
        d_new_hashes = temp;
        
        current_n = (current_n + 1) / 2; // Round up for odd number of hashes
    }
    
    // Copy the final hash (merkle root) back to host
    cudaMemcpy(h_hashes, d_hashes, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
    memcpy(merkle_root, h_hashes, SHA256_HASH_SIZE);
    
    // Free memory
    free(h_hashes);
    cudaFree(d_transactions);
    cudaFree(d_hashes);
    cudaFree(d_new_hashes);
}

// CUDA kernel to find a valid nonce
__global__ void find_nonce_kernel(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, 
                                 size_t current_length, BYTE *block_hash, uint32_t *found_nonce,
                                 int *found_flag, uint32_t start_nonce, uint32_t nonces_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce_start = start_nonce + tid * nonces_per_thread;
    uint32_t nonce_end = min(nonce_start + nonces_per_thread, max_nonce);
    
    // Create a local copy of block_content to avoid race conditions
    BYTE local_block[BLOCK_SIZE];
    d_strcpy((char *)local_block, (const char *)block_content);
    
    BYTE local_hash[SHA256_HASH_SIZE];
    char nonce_string[NONCE_SIZE];
    
    for (uint32_t nonce = nonce_start; nonce < nonce_end; nonce++) {
        // Check if another thread has already found a valid nonce
        if (*found_flag)
            return;
            
        // Convert nonce to string using device function
        intToString(nonce, nonce_string);
        
        // Copy nonce to the end of the block content
        d_strcpy((char *)local_block + current_length, nonce_string);
        
        // Compute hash
        apply_sha256(local_block, local_hash);
        
        // Check if hash meets difficulty requirement
        if (compare_hashes(local_hash, difficulty) <= 0) {
            // Found a valid nonce
            *found_nonce = nonce;
            *found_flag = 1;
            
            // Copy the hash to output
            for (int i = 0; i < SHA256_HASH_SIZE; i++) {
                block_hash[i] = local_hash[i];
            }
            return;
        }
    }
}

int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    // Allocate device memory
    BYTE *d_difficulty, *d_block_content, *d_block_hash;
    uint32_t *d_found_nonce;
    int *d_found_flag;
    
    cudaMalloc((void **)&d_difficulty, SHA256_HASH_SIZE);
    cudaMalloc((void **)&d_block_content, BLOCK_SIZE);
    cudaMalloc((void **)&d_block_hash, SHA256_HASH_SIZE);
    cudaMalloc((void **)&d_found_nonce, sizeof(uint32_t));
    cudaMalloc((void **)&d_found_flag, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_content, block_content, current_length, cudaMemcpyHostToDevice);
    
    // Initialize found flag to 0
    int found_flag = 0;
    cudaMemcpy(d_found_flag, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int maxBlocks = 65535; // Maximum number of blocks
    
    // Calculate how many nonces each thread should check
    uint32_t total_threads = threadsPerBlock * maxBlocks;
    uint64_t nonces_per_thread = (uint64_t)(max_nonce + (uint64_t)total_threads - 1) / total_threads;
    
    // Adjust number of blocks if we have fewer nonces than threads
    int blocksPerGrid = min(
        maxBlocks,
        (int)(((uint64_t)max_nonce + threadsPerBlock - 1) / threadsPerBlock)
    );
    
    
    // Launch kernel
    find_nonce_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_difficulty, max_nonce, d_block_content, current_length, 
        d_block_hash, d_found_nonce, d_found_flag, 0, nonces_per_thread);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Check if a valid nonce was found
    cudaMemcpy(&found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
    
    int result = 1; // Default: no valid nonce found
    
    if (found_flag) {
        // Copy the valid nonce and hash back to host
        cudaMemcpy(valid_nonce, d_found_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_hash, d_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
        result = 0; // Success
    }
    
    // Free device memory
    cudaFree(d_difficulty);
    cudaFree(d_block_content);
    cudaFree(d_block_hash);
    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);
    
    return result;
}

__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu() {
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);
}
