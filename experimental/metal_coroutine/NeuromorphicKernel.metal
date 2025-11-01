//  NeuromorphicKernel.metal
//  MetalCoroutinesTest
//
//  Created by Sydney Bach on 2/23/25.

#include <metal_stdlib>
using namespace metal;

// Threadgroup size - adjust as needed
#define TILE_SIZE 16

// Structure for scalar parameters
struct KernelParams {
    int N;
    float dt;
    int num_steps;
    float alpha;
    float target_sum;
    float neural_clock;
    uint step_num;
    float eta;
    bool use_hebbian;
    float decay_rate;
};

kernel void liquid_cfc_xlstm_kernel(
                                    uint2 gid [[thread_position_in_grid]],
                                    constant KernelParams &params [[buffer(0)]],
                                    device atomic_float* W_recurrent [[buffer(1)]],
                                    constant float* W_i [[buffer(2)]],
                                    constant float* U_i [[buffer(3)]],
                                    constant float* b_i [[buffer(4)]],
                                    constant float* W_f [[buffer(5)]],
                                    constant float* U_f [[buffer(6)]],
                                    constant float* b_f [[buffer(7)]],
                                    constant float* W_o [[buffer(8)]],
                                    constant float* U_o [[buffer(9)]],
                                    constant float* b_o [[buffer(10)]],
                                    constant float* W_g [[buffer(11)]],
                                    constant float* U_g [[buffer(12)]],
                                    constant float* b_g [[buffer(13)]],
                                    constant float* lambda [[buffer(14)]],
                                    constant int* gate_mask [[buffer(15)]],
                                    constant int* lambda_mask [[buffer(16)]],
                                    device float* h_liquid_read [[buffer(17)]],
                                    device float* h_liquid_write [[buffer(18)]],
                                    device float* c_t [[buffer(19)]],
                                    device float* n_t [[buffer(20)]],
                                    device char* logBuffer [[buffer(21)]],
                                    uint2 lid [[thread_position_in_threadgroup]]
                                    ) {
    uint i = gid.y * TILE_SIZE + lid.y;
    if (i >= uint(params.N)) return;

    device float* h_liquid_current = (params.step_num % 2u == 0u) ? h_liquid_read : h_liquid_write;
    device float* h_liquid_next = (params.step_num % 2u == 0u) ? h_liquid_write : h_liquid_read;

    threadgroup float W_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float h_tile[TILE_SIZE];

    float x_t = 0.0f;
    uint numTiles = (uint(params.N) + TILE_SIZE - 1u) / TILE_SIZE;
    for (uint tile = 0; tile < numTiles; tile++) {
        uint row = gid.y * TILE_SIZE + lid.y;
        uint col = tile * TILE_SIZE + lid.x;
        if (row < uint(params.N) && col < uint(params.N)) {
            W_tile[lid.y][lid.x] = atomic_load_explicit(&W_recurrent[row * uint(params.N) + col], memory_order_relaxed);
        } else {
            W_tile[lid.y][lid.x] = 0.0f;
        }

        uint h_index = tile * TILE_SIZE + lid.x;
        if (h_index < uint(params.N)) {
            h_tile[lid.x] = h_liquid_current[h_index];
        } else {
            h_tile[lid.x] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            if (row < uint(params.N) && (tile * TILE_SIZE + k) < uint(params.N)) {
                x_t += W_tile[lid.y][k] * h_tile[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float i_t, f_t, o_t;
    if (gate_mask[i] == 0) {
        i_t = 1.0f; f_t = 1.0f; o_t = 1.0f;
    } else {
        float input_i = W_i[i] * x_t + U_i[i] * h_liquid_current[i] + b_i[i] - n_t[i];
        float input_f = W_f[i] * x_t + U_f[i] * h_liquid_current[i] + b_f[i] - n_t[i];
        float input_o = W_o[i] * x_t + U_o[i] * h_liquid_current[i] + b_o[i] - n_t[i];
        i_t = exp(input_i);
        f_t = exp(input_f);
        o_t = exp(input_o);
    }

    float g_t = 1.0f / (1.0f + exp(-(W_g[i] * x_t + U_g[i] * h_liquid_current[i] + b_g[i])));
    float c_new = f_t * c_t[i] + i_t * g_t;
    float feed_forward = o_t * (1.0f / (1.0f + exp(-c_new)));
    float effective_lambda = (lambda_mask[i] == 0) ? 0.0f : lambda[i];
    float h_old = h_liquid_current[i];
    float denom = 1.0f + params.neural_clock * effective_lambda;
    float h_new = (h_old + params.neural_clock * feed_forward) / denom;

    if (gate_mask[i] == 1) {
        float sum_gates = i_t + f_t + o_t;
        float n_new = n_t[i] + params.alpha * (sum_gates - params.target_sum);
        n_t[i] = n_new;
    }

    if (params.use_hebbian) {
        for (uint j = 0; j < uint(params.N); j++) {
            float delta_w = params.eta * h_liquid_next[j] * h_new * i_t;
            float w_recurrent_value = atomic_load_explicit(&W_recurrent[j * uint(params.N) + i], memory_order_relaxed);
            delta_w -= params.decay_rate * w_recurrent_value;
            atomic_fetch_add_explicit((device atomic_float*)&W_recurrent[j * uint(params.N) + i], delta_w, memory_order_relaxed);
        }
    }

    h_liquid_next[i] = h_new;
    c_t[i] = c_new;

    if (i == 0) {
        constant char* logMessage = "Neuron 0 state: ";
        int logMessageLength = 0;
        while (logMessage[logMessageLength] != '\0') { logMessageLength++; }
        for (int k = 0; k < logMessageLength; ++k) { logBuffer[k] = logMessage[k]; }
        int index = logMessageLength;
        float value = h_new; int intPart = int(value); float fracPart = value - float(intPart); int fracInt = int(fracPart * 1000000);
        if (intPart == 0) { logBuffer[index++] = '0'; }
        else {
            if (intPart < 0) { logBuffer[index++] = '-'; intPart = -intPart; }
            char intStr[10]; int intLen = 0; while (intPart > 0) { intStr[intLen++] = '0' + (intPart % 10); intPart /= 10; }
            for (int j = intLen - 1; j >= 0; --j) { logBuffer[index++] = intStr[j]; }
        }
        logBuffer[index++] = '.';
        char fracStr[7]; int fracLen = 0; while (fracInt > 0) { fracStr[fracLen++] = '0' + (fracInt % 10); fracInt /= 10; }
        for (int j = 6 - fracLen; j > 0; --j) { logBuffer[index++] = '0'; }
        for (int j = fracLen - 1; j >= 0; --j) { logBuffer[index++] = fracStr[j]; }
        logBuffer[index] = '\0';
    }
}

