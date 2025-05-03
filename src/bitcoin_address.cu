#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <secp256k1.h>
#include <time.h>
#include "GPU/GPUStream.h"
#include "sha256_cuda.cuh"
#include "ripemd160_cuda.cuh"
#include "GPU/GPUConstants.h"
#include "CPU/Int.h"
#include "CPU/Point.h"
#include "CPU/SECP256k1.h"
#include "GPU/GPUSecp.h"

// Estrutura para armazenar os parâmetros de busca
struct SearchParams {
    unsigned char private_key[32];
    unsigned char target_address[20];
    int indices[12];
    int num_indices;
};

// Função auxiliar para converter hex string para bytes
bool hex_to_bytes(const char* hex_str, unsigned char* bytes, size_t length) {
    if (strlen(hex_str) != length * 2) return false;
    
    for (size_t i = 0; i < length; i++) {
        char hex_byte[3] = {hex_str[i*2], hex_str[i*2+1], 0};
        char* end_ptr;
        bytes[i] = (unsigned char)strtol(hex_byte, &end_ptr, 16);
        if (*end_ptr != 0) return false;
    }
    return true;
}

// Função para formatar a velocidade
const char* formatSpeed(double speed) {
    static char buffer[32];
    if (speed >= 1e9) {
        snprintf(buffer, sizeof(buffer), "%.2f Gkeys/s", speed / 1e9);
    } else if (speed >= 1e6) {
        snprintf(buffer, sizeof(buffer), "%.2f Mkeys/s", speed / 1e6);
    } else if (speed >= 1e3) {
        snprintf(buffer, sizeof(buffer), "%.2f Kkeys/s", speed / 1e3);
    } else {
        snprintf(buffer, sizeof(buffer), "%.2f keys/s", speed);
    }
    return buffer;
}

// Variáveis globais para tabelas G na GPU
uint8_t *d_gTableX = nullptr;
uint8_t *d_gTableY = nullptr;

// Função para carregar as tabelas G
void loadGTable(uint8_t *gTableX, uint8_t *gTableY) {
    // Alocar memória temporária na CPU
    uint8_t *hostTableX = new uint8_t[NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT];
    uint8_t *hostTableY = new uint8_t[NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT];

    // Gerar tabelas na CPU
    Secp256K1 *secp = new Secp256K1();
    secp->Init();

    for (int i = 0; i < NUM_GTABLE_CHUNK; i++) {
        for (int j = 0; j < NUM_GTABLE_VALUE - 1; j++) {
            int element = (i * NUM_GTABLE_VALUE) + j;
            Point p = secp->GTable[element];
            for (int b = 0; b < 32; b++) {
                hostTableX[(element * SIZE_GTABLE_POINT) + b] = p.x.GetByte64(b);
                hostTableY[(element * SIZE_GTABLE_POINT) + b] = p.y.GetByte64(b);
            }
        }
    }

    delete secp;

    cudaError_t err;
    err = cudaMemcpy(gTableX, hostTableX, 
                     NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT, 
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Erro ao copiar gTableX para GPU: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(gTableY, hostTableY, 
                     NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT, 
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Erro ao copiar gTableY para GPU: %s\n", cudaGetErrorString(err));
    }

    delete[] hostTableX;
    delete[] hostTableY;
}

void freeGPUTables() {
    if (d_gTableX) cudaFree(d_gTableX);
    if (d_gTableY) cudaFree(d_gTableY);
    d_gTableX = nullptr;
    d_gTableY = nullptr;
}

bool initGPUTables() {
    cudaError_t err;
    size_t tableSize = NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE * SIZE_GTABLE_POINT;
    printf("Alocando %zu bytes para cada tabela...\n", tableSize);

    err = cudaMalloc(&d_gTableX, tableSize);
    if (err != cudaSuccess) {
        printf("Erro ao alocar memória para gTableX: %s\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc(&d_gTableY, tableSize);
    if (err != cudaSuccess) {
        printf("Erro ao alocar memória para gTableY: %s\n", cudaGetErrorString(err));
        cudaFree(d_gTableX);
        return false;
    }

    printf("Carregando tabelas...\n");
    loadGTable(d_gTableX, d_gTableY);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro após carregar tabelas: %s\n", cudaGetErrorString(err));
        freeGPUTables();
        return false;
    }

    printf("Tabelas carregadas com sucesso!\n");
    return true;
}

// Função device para incrementar a chave privada
__device__ void increment_private_key_gpu(unsigned char *private_key, uint64_t increment) {
    uint64_t carry = increment;
    for (int i = 31; i >= 0 && carry > 0; i--) {
        uint64_t sum = (uint64_t)private_key[i] + carry;
        private_key[i] = sum & 0xFF;
        carry = sum >> 8;
    }
}

// Função device para aplicar permutação em índices específicos
__device__ void apply_permutation(unsigned char* private_key, const int* indices, int num_indices, uint64_t permutation) {
    // Cada índice trabalha com 4 bits (um dígito hexadecimal)
    for (int i = 0; i < num_indices; i++) {
        int index = num_indices - 1 - i; // Inverter a ordem dos índices (da direita para a esquerda)
        int byte_pos = indices[index] / 2;  // Posição do byte (cada byte tem 2 dígitos hex)
        int is_high_nibble = indices[index] % 2 == 0;  // Se é o nibble alto do byte
        
        // Extrai o dígito da permutação
        unsigned char digit = (permutation >> (i * 4)) & 0xF;
        
        // Atualiza o byte correto mantendo o outro nibble inalterado
        if (is_high_nibble) {
            private_key[byte_pos] = (private_key[byte_pos] & 0x0F) | (digit << 4);
        } else {
            private_key[byte_pos] = (private_key[byte_pos] & 0xF0) | digit;
        }
    }
}

// Kernel otimizado para processar múltiplas chaves em paralelo com suporte a índices específicos
__global__ void bitcoin_address_kernel(unsigned char* base_private_key, unsigned char* bitcoin_address, int* match_found, const unsigned char* target_address, uint64_t start_permutation, uint64_t permutations_per_thread, const int* indices, int num_indices, uint8_t* d_gTableX, uint8_t* d_gTableY) {
    __shared__ uint8_t shared_gTableX[1024]; // Ajustar tamanho com base no uso
    __shared__ uint8_t shared_gTableY[1024];
    __shared__ unsigned char shared_target_address[RIPEMD160_DIGEST_SIZE];

    // Carregar dados na memória compartilhada
    if (threadIdx.x == 0) {
        memcpy(shared_target_address, target_address, RIPEMD160_DIGEST_SIZE);
    }
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        shared_gTableX[i] = d_gTableX[i];
        shared_gTableY[i] = d_gTableY[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char local_private_key[32];
    
    // Copiar chave base para memória local
    for (int i = 0; i < 32; i++) {
        local_private_key[i] = base_private_key[i];
    }
    

    
    unsigned char sha256_hash[SHA256_DIGEST_SIZE];
    unsigned char ripemd160_hash[RIPEMD160_DIGEST_SIZE];
    unsigned char public_key[33];  // Compressed public key format
    
    // Processa todas as permutações atribuídas a esta thread
    for(uint64_t perm = 0; perm < permutations_per_thread && !(*match_found); perm++) {
        // Valor para incremento baseado na permutação atual
        uint64_t current_perm = start_permutation + (idx * permutations_per_thread) + perm;
        
        // Aplica a permutação atual à chave privada local
        apply_permutation(local_private_key, indices, num_indices, current_perm);
        
        // Converter a chave privada para o formato correto
        uint16_t privKeyChunks[NUM_GTABLE_CHUNK] = {0};
        uint16_t* privKeyShorts = (uint16_t*)local_private_key;
        
        for (int j = 0; j < 16; j++) {
            uint16_t value = privKeyShorts[j];
            value = ((value & 0xFF00) >> 8) | ((value & 0x00FF) << 8);
            privKeyChunks[15 - j] = value;
        }

        // Gerar public key usando as tabelas G
        uint64_t pubX[4], pubY[4];
        _PointMultiSecp256k1(pubX, pubY, privKeyChunks, d_gTableX, d_gTableY, shared_gTableX, shared_gTableY);

        // Converter para formato comprimido (33 bytes)
        public_key[0] = 0x02 | (pubY[0] & 1);
        for (int j = 0; j < 32; j++) {
            public_key[j+1] = ((unsigned char*)pubX)[31-j];
        }
        
        // Calcular hash SHA256 da chave pública
        sha256_gpu(public_key, 33, sha256_hash);
        
        // Calcular hash RIPEMD160 do hash SHA256
        ripemd160_gpu(sha256_hash, SHA256_DIGEST_SIZE, ripemd160_hash);



        // Verificar match
        bool match = true;
        for (int j = 0; j < RIPEMD160_DIGEST_SIZE; j++) {
            if (ripemd160_hash[j] != shared_target_address[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            atomicExch(match_found, 1);
            memcpy(bitcoin_address, ripemd160_hash, RIPEMD160_DIGEST_SIZE);
            memcpy(base_private_key, local_private_key, 32);
            printf("Correspondência Encontrada na Thread %d, Permutação %llu!\n", idx, current_perm);
        }
    }
}

int main(int argc, char **argv) {
    int blockSize = 256;  // default
    int numBlocks = 2048; // aumentado
    int numStreams = 8; // Limita a 4 streams para a 4060   // default
    int keysPerThread = 10; // Número de chaves por thread
    // const int BATCH_SIZE = 32; // Commented out as it is declared but never referenced
    
    // Inicializar parâmetros de busca
    SearchParams params = {0};
    memset(params.private_key, 0, 32);
    memset(params.target_address, 0, 20);
    params.num_indices = 0;
    
    // Variável para controlar uso de índices específicos
    // bool use_specific_indices = false; // Commented out as it is set but never used

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-block") == 0 && i + 1 < argc) {
            blockSize = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-grid") == 0 && i + 1 < argc) {
            numBlocks = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-streams") == 0 && i + 1 < argc) {
            numStreams = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-kpt") == 0 && i + 1 < argc) {
            keysPerThread = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            if (strlen(argv[i + 1]) != 64) {
                printf("Erro: chave privada deve ter 64 dígitos hexadecimais\n");
                return 1;
            }
            if (!hex_to_bytes(argv[i + 1], params.private_key, 32)) {
                printf("Erro: chave privada inválida\n");
                return 1;
            }
            i++;
        }
        else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            if (strlen(argv[i + 1]) != 40) { // hash160 em hex tem 40 caracteres
                printf("Erro: endereço Bitcoin (hash160) deve ter 40 dígitos hexadecimais\n");
                return 1;
            }
            if (!hex_to_bytes(argv[i + 1], params.target_address, 20)) {
                printf("Erro: endereço Bitcoin inválido\n");
                return 1;
            }
            i++;
        }
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            // Índices para brute force (formato: 1,2,3,4 ou ranges como 1-4)
            char* token = strtok(argv[i + 1], ",");
            while (token != NULL && params.num_indices < 12) {
                // Verificar se é um range (ex: 1-4)
                char* range_separator = strchr(token, '-');
                if (range_separator) {
                    *range_separator = '\0'; // Separar a string
                    int start = atoi(token);
                    int end = atoi(range_separator + 1);
                    
                    // Ajustar para índices 0-63 (usuário fornece 1-64)
                    start = (start > 0) ? start - 1 : 0;
                    end = (end > 0) ? end - 1 : 0;
                    
                    // Limitar aos valores válidos
                    start = (start < 0) ? 0 : (start > 63) ? 63 : start;
                    end = (end < 0) ? 0 : (end > 63) ? 63 : end;
                    
                    // Adicionar todos os índices do range
                    for (int idx = start; idx <= end && params.num_indices < 12; idx++) {
                        params.indices[params.num_indices++] = idx;
                    }
                } else {
                    // Índice individual
                    int idx = atoi(token);
                    // Ajustar para índices 0-63 (usuário fornece 1-64)
                    idx = (idx > 0) ? idx - 1 : 0;
                    // Limitar aos valores válidos
                    idx = (idx < 0) ? 0 : (idx > 63) ? 63 : idx;
                    params.indices[params.num_indices++] = idx;
                }
                token = strtok(NULL, ",");
            }
            // use_specific_indices = (params.num_indices > 0); // Commented out as it is set but never used
        }
        else if (strcmp(argv[i], "-private") == 0 && i + 1 < argc) {
            if (!hex_to_bytes(argv[i + 1], params.private_key, 32)) {
                printf("Erro: chave privada inválida. Use 64 caracteres hexadecimais\n");
                return 1;
            }
            i++;
        }
    }

    // Validar parâmetros
    if (params.num_indices == 0) {
        printf("Erro: índices não especificados. Use -i para especificar os índices\n");
        return 1;
    }

    printf("Inicializando tabelas G na GPU...\n");
    if (!initGPUTables()) {
        printf("Falha ao inicializar tabelas G. Abortando.\n");
        return 1;
    }

    // Calcular o número total de permutações possíveis
    uint64_t total_permutations = 1ULL;
    bool overflow = false;
    for (int i = 0; i < params.num_indices; i++) {
        if (total_permutations > (UINT64_MAX / 16)) {
            overflow = true;
            break;
        }
        total_permutations *= 16; // Cada dígito hex tem 16 possibilidades
    }
    if (overflow) {
        printf("Aviso: Número de permutações muito grande, possível overflow detectado.\n");
        total_permutations = UINT64_MAX;
    }
    printf("Total de permutações a testar: %lu\n", total_permutations);

    // Batch size optimization: Adjust based on total threads (~1000 permutations per thread)
    uint64_t batch_size = min(total_permutations, (uint64_t)blockSize * numBlocks * 1000);
    // Ensure even distribution across streams to avoid imbalance
    uint64_t perms_per_stream = batch_size / numStreams;
    uint64_t processed_permutations = 0;
    uint64_t total_addresses_processed = 0;
    int current_stream = 0;

    printf("Iniciando busca com batch size de %lu permutações...\n", batch_size);
    clock_t search_start_time = clock();
    clock_t start_time = clock();
    uint64_t addresses_processed = 0;
    // double display_interval = 1.0; // Commented out as it is declared but never referenced
    
    // Copiar índices para a GPU
    int* d_indices;
    cudaMalloc(&d_indices, params.num_indices * sizeof(int));
    cudaMemcpy(d_indices, params.indices, params.num_indices * sizeof(int), cudaMemcpyHostToDevice);
    
    // Alocar memória pinned na CPU para transferências mais rápidas
    unsigned char *host_private_key;
    unsigned char *host_bitcoin_address;
    cudaHostAlloc(&host_private_key, 32, cudaHostAllocDefault);
    cudaHostAlloc(&host_bitcoin_address, RIPEMD160_DIGEST_SIZE, cudaHostAllocDefault);
    memcpy(host_private_key, params.private_key, 32);

    // Alocar memória para o endereço target (compartilhado entre streams)
    unsigned char *d_target_address;
    cudaMalloc(&d_target_address, RIPEMD160_DIGEST_SIZE);

    const int NUM_STREAMS = numStreams;
    CUDAStream streams[NUM_STREAMS];

    // Inicializar streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i].stream);
        cudaMalloc(&streams[i].d_private_key, 32);
        cudaMalloc(&streams[i].d_bitcoin_address, RIPEMD160_DIGEST_SIZE);
        cudaMalloc(&streams[i].d_match_found, sizeof(int));
        streams[i].in_use = false;
    }

    cudaMemcpyAsync(d_target_address, params.target_address, 
                    RIPEMD160_DIGEST_SIZE, cudaMemcpyHostToDevice, streams[0].stream);

    // Copiar a chave privada base para o dispositivo (uma vez por stream)
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(streams[i].d_private_key, params.private_key,
                        32, cudaMemcpyHostToDevice, streams[i].stream);
    }

    const uint64_t TOTAL_THREADS = blockSize * numBlocks;
    uint64_t permutations_per_thread;
    int match_found_host = 0;
    
    while (processed_permutations < total_permutations && !match_found_host) {
        // Verificar se há streams disponíveis
        for (int i = 0; i < NUM_STREAMS; i++) {
            if (streams[i].in_use) {
                cudaError_t err = cudaStreamQuery(streams[i].stream);
                if (err == cudaSuccess) {
                    // Stream terminou, verificar resultado
                    cudaMemcpyAsync(&match_found_host, streams[i].d_match_found,
                                  sizeof(int), cudaMemcpyDeviceToHost, streams[i].stream);
                    cudaStreamSynchronize(streams[i].stream);
                    if (match_found_host) {
                        current_stream = i;
                        printf("Correspondência detectada no stream %d!\n", i);
                        // Copiar a chave privada encontrada de volta para o host
                        unsigned char found_private_key[32];
                        cudaMemcpy(found_private_key, streams[i].d_private_key, 32, cudaMemcpyDeviceToHost);
                        // Exibir a chave encontrada em verde e letras grandes (usando ANSI escape codes)
                        printf("\033[1;32m===========================================\n");
                        printf("CHAVE PRIVADA ENCONTRADA!!!\n");
                        printf("Chave: ");
                        for (int k = 0; k < 32; k++) {
                            printf("%02x", found_private_key[k]);
                        }
                        printf("\n===========================================\033[0m\n");
                        // Salvar a chave em ACHADOS.txt
                        FILE *file = fopen("ACHADOS.txt", "a");
                        if (file) {
                            fprintf(file, "Chave Privada Encontrada: ");
                            for (int k = 0; k < 32; k++) {
                                fprintf(file, "%02x", found_private_key[k]);
                            }
                            fprintf(file, "\n");
                            fclose(file);
                            printf("Chave salva em ACHADOS.txt\n");
                        } else {
                            printf("Erro ao salvar chave em ACHADOS.txt\n");
                        }
                        break;
                    }
                    streams[i].in_use = false;
                } else if (err != cudaErrorNotReady) {
                    printf("Erro no stream %d: %s\n", i, cudaGetErrorString(err));
                    streams[i].in_use = false;
                }
            }
        }
        if (match_found_host) break;
        
        if (!streams[current_stream].in_use) {
            // Stream está disponível, configurar novo batch
            uint64_t start_permutation = processed_permutations;
            if (start_permutation >= total_permutations) {
                break; // We've processed all permutations
            }
            uint64_t remaining_permutations = total_permutations - start_permutation;
            batch_size = (remaining_permutations < 75000000ULL) ? remaining_permutations : 75000000ULL;
            permutations_per_thread = (batch_size + TOTAL_THREADS - 1) / TOTAL_THREADS;
            
            bitcoin_address_kernel<<<numBlocks, blockSize, 0, streams[current_stream].stream>>>(
                streams[current_stream].d_private_key,
                streams[current_stream].d_bitcoin_address,
                streams[current_stream].d_match_found,
                d_target_address,
                start_permutation,
                permutations_per_thread,
                d_indices,
                params.num_indices,
                d_gTableX,
                d_gTableY
            );
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Erro no kernel: %s\n", cudaGetErrorString(err));
            }

            streams[current_stream].in_use = true;

            addresses_processed += batch_size;
            processed_permutations += batch_size;
            total_addresses_processed += batch_size;

            // Estatísticas de performance - exibir a cada novo batch
            clock_t current_time = clock();
            double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            double keys_per_second = addresses_processed / elapsed_time;
            printf("\rVelocidade: %s | Total processado: %lu", 
                   formatSpeed(keys_per_second), 
                   total_addresses_processed);
            fflush(stdout);
            start_time = clock();
            addresses_processed = 0;
        }
    }

    // Após o loop principal, garantir que todos os streams ativos sejam sincronizados antes de concluir
    bool any_stream_active = true;
    while (any_stream_active && !match_found_host) {
        any_stream_active = false;
        for (int i = 0; i < NUM_STREAMS; i++) {
            if (streams[i].in_use) {
                any_stream_active = true;
                cudaError_t err = cudaStreamQuery(streams[i].stream);
                if (err == cudaSuccess) {
                    cudaMemcpyAsync(&match_found_host, streams[i].d_match_found,
                                  sizeof(int), cudaMemcpyDeviceToHost, streams[i].stream);
                    cudaStreamSynchronize(streams[i].stream);
                    if (match_found_host) {
                        current_stream = i;
                        printf("Correspondência detectada no stream %d!\n", i);
                        // Copiar a chave privada encontrada de volta para o host
                        unsigned char found_private_key[32];
                        cudaMemcpy(found_private_key, streams[i].d_private_key, 32, cudaMemcpyDeviceToHost);
                        // Exibir a chave encontrada em verde e letras grandes (usando ANSI escape codes)
                        printf("\033[1;32m===========================================\n");
                        printf("CHAVE PRIVADA ENCONTRADA!!!\n");
                        printf("Chave: ");
                        for (int k = 0; k < 32; k++) {
                            printf("%02x", found_private_key[k]);
                        }
                        printf("\n===========================================\033[0m\n");
                        // Salvar a chave em ACHADOS.txt
                        FILE *file = fopen("ACHADOS.txt", "a");
                        if (file) {
                            fprintf(file, "Chave Privada Encontrada: ");
                            for (int k = 0; k < 32; k++) {
                                fprintf(file, "%02x", found_private_key[k]);
                            }
                            fprintf(file, "\n");
                            fclose(file);
                            printf("Chave salva em ACHADOS.txt\n");
                        } else {
                            printf("Erro ao salvar chave em ACHADOS.txt\n");
                        }
                        break;
                    }
                    streams[i].in_use = false;
                } else if (err != cudaErrorNotReady) {
                    printf("Erro no stream %d: %s\n", i, cudaGetErrorString(err));
                    streams[i].in_use = false;
                }
            }
        }
    }

    clock_t search_end_time = clock();
    double total_search_time = (double)(search_end_time - search_start_time) / CLOCKS_PER_SEC;
    double avg_speed = total_addresses_processed / total_search_time;

    if (!match_found_host) {
        printf("\n\033[31mChave não encontrada após testar todas as %lu permutações (%.2f segundos)\033[0m\n", 
               total_permutations, total_search_time);
    }
    
    // Cleanup
    cudaFreeHost(host_private_key);
    cudaFreeHost(host_bitcoin_address);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i].stream);
        cudaFree(streams[i].d_private_key);
        cudaFree(streams[i].d_bitcoin_address);
        cudaFree(streams[i].d_match_found);
        cudaStreamDestroy(streams[i].stream);
    }
    cudaFree(d_target_address);
    freeGPUTables();

    return 0;
}