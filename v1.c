#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef struct {
    // Filters (weights)
    float *filter_1;         // 16 × 1 × 3 × 3 = 144 elements
    float *filter_2;         // 32 × 16 × 3 × 3 = 4,608 elements  
    float *filter_3;         // 64 × 32 × 3 × 3 = 18,432 elements
    float *filter_4;         // 32 × 64 × 3 × 3 = 18,432 elements (transpose)
    float *filter_41;        // 32 × 32 × 3 × 3 = 9,216 elements
    float *filter_5;         // 16 × 32 × 3 × 3 = 4,608 elements (transpose)
    float *filter_51;        // 16 × 16 × 3 × 3 = 2,304 elements
    float *filter_6;         // 1 × 16 × 1 × 1 = 16 elements

    // Biases
    float *bias_1;           // 16 elements
    float *bias_2;           // 32 elements
    float *bias_3;           // 64 elements
    float *bias_4;           // 32 elements
    float *bias_41;          // 32 elements
    float *bias_5;           // 16 elements
    float *bias_51;          // 16 elements
    float *bias_6;           // 1 element

    // Time embedding weights
    float *matmul_1;         // 1 × TIME_HIDDEN_SIZE elements
    float *matmul_2;         // TIME_HIDDEN_SIZE × TIME_HIDDEN_SIZE elements
    float *bias_m2;          // TIME_HIDDEN_SIZE elements

    // Gradient arrays for filters
    float *grad_filter_1;    // 16 × 1 × 3 × 3 = 144 elements
    float *grad_filter_2;    // 32 × 16 × 3 × 3 = 4,608 elements
    float *grad_filter_3;    // 64 × 32 × 3 × 3 = 18,432 elements
    float *grad_filter_4;    // 32 × 64 × 3 × 3 = 18,432 elements
    float *grad_filter_41;   // 32 × 32 × 3 × 3 = 9,216 elements
    float *grad_filter_5;    // 16 × 32 × 3 × 3 = 4,608 elements
    float *grad_filter_51;   // 16 × 16 × 3 × 3 = 2,304 elements
    float *grad_filter_6;    // 1 × 16 × 1 × 1 = 16 elements

    // Gradient arrays for biases
    float *grad_bias_1;      // 16 elements
    float *grad_bias_2;      // 32 elements
    float *grad_bias_3;      // 64 elements
    float *grad_bias_4;      // 32 elements
    float *grad_bias_41;     // 32 elements
    float *grad_bias_5;      // 16 elements
    float *grad_bias_51;     // 16 elements
    float *grad_bias_6;      // 1 element

    // Gradient arrays for time embeddings
    float *grad_matmul_1;    // 1 × TIME_HIDDEN_SIZE elements
    float *grad_matmul_2;    // TIME_HIDDEN_SIZE × TIME_HIDDEN_SIZE elements
    float *grad_bias_m2;     // TIME_HIDDEN_SIZE elements
} Network;

double get_time_diff(struct timespec start, struct timespec end){
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

#define INPUT_SIZE 784
#define TIME_HIDDEN_SIZE 64
#define OUTPUT_SIZE 784
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define CHANNEL_1 16
#define CHANNEL_2 32
#define CHANNEL_3 64
#define HEIGHT 28
#define WIDTH 28


#define TIMESTEPS 1000
#define BETA_1 1e-4
#define BETA_2 0.02


void load_data(const char *filename, float *data, int size){
    FILE *file = fopen(filename, "rb");
    if (file == NULL){
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size){
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
    }
    fclose(file);
}


void initialize_weights(float *weights, int c1, int c2, int height, int width){
    int total_1 = c2 * height * width;  
    int total_2 = c1 * c2 * height * width;  
    
    float scale = sqrtf(2.0f / total_1);  
    
    for (int i = 0; i < total_2; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;  
    }
}

void initialize_matmul_weights(float *weights, int input_size, int output_size) {
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

void initialize_bias(float *biases, int size){
    for (int i = 0; i < size; i++) {
        biases[i] = 0.0f;
    }
}


void normalize_data(float *data, int size){
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++){
        data[i] = (data[i] - mean) / std;
    }
}

void matmul_a_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}

void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < m; l++) {
                C[i * k + j] += A[l * n + i] * B[l * k + j];
            }
        }
    }
}

void conv2d_forward(float *input, float *output, float *filter, int r, int in_channels, int out_channels, int height, int width){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < out_channels; j++){
            for (int k = 0; k < height; k++){
                for (int l = 0; l < width; l++){
                    float Pval = 0.0;
                    for (int in_ch = 0; in_ch < in_channels; in_ch++){
                        for (int m = 0; m < 2*r+1; m++){
                            for (int n = 0; n < 2*r+1; n++){
                                int inRow = k - r + m;
                                int inCol = l - r + n;
                                if (inRow >=0 && inRow < height && inCol >=0 && inCol <width){
                                    int filter_idx = j * (in_channels * (2*r+1) * (2*r+1)) + in_ch * ((2*r+1) * (2*r+1)) + m * (2*r+1) + n;
                                    int input_idx = i * (in_channels * height * width) + in_ch * (height * width) + inRow * width + inCol;
                                    Pval += filter[filter_idx] * input[input_idx];
                                }
                            }
                        }

                    }        
                    int output_idx = i * (out_channels * height * width) + j * (height * width) + k * width + l;
                    output[output_idx] = Pval;
                }
            }
        }
    }
}

void conv2d_backward(float *input, float *output, float *filter, float *grad_input, float *grad_output, float *grad_filter, int r, int in_channels, int out_channels, int height, int width){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int in_ch = 0; in_ch < in_channels; in_ch++){
            for (int k = 0; k < height; k++){
                for (int l = 0; l < width; l++){
                    float grad_sum = 0.0f;
                    for (int out_ch = 0; out_ch < out_channels; out_ch++){
                        for (int m = 0; m < 2*r+1; m++){
                            for (int n = 0; n < 2*r+1; n++){
                                int outRow = k + r - m;
                                int outCol = l + r - n;
                                if (outRow >= 0 && outRow < height && outCol >= 0 && outCol < width) {
                                    int filter_idx = out_ch * (in_channels * (2*r+1) * (2*r+1)) + in_ch * ((2*r+1) * (2*r+1)) + (2*r-m) * (2*r+1) + (2*r-n);
                                    int output_idx = i * (out_channels * height * width) + out_ch * (height * width) + outRow * width + outCol;
                                    grad_sum += filter[filter_idx] * grad_output[output_idx];
                                }
                            }
                        }
                    }
                    int input_idx = i * (in_channels * height * width) + in_ch * (height * width) + k * width + l;
                    grad_input[input_idx] = grad_sum;
                }
            }
        }
    }

    for (int out_ch = 0; out_ch < out_channels; out_ch++){
        for (int in_ch = 0; in_ch < in_channels; in_ch++){
            for (int m = 0; m < 2*r+1; m++){
                for (int n = 0; n < 2*r+1; n++){
                    float grad_sum = 0.0f;
                    for (int i = 0; i < BATCH_SIZE; i++){
                        for (int k = 0; k < height; k++){
                            for (int l = 0; l < width; l++){
                                int inRow = k + r - m;
                                int inCol = l + r - n;
                                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                                    int input_idx = i * (in_channels * height * width) + in_ch * (height * width) + inRow * width + inCol;
                                    int output_idx = i * (out_channels * height * width) + out_ch * (height * width) + k * width + l;
                                    grad_sum += input[input_idx] * grad_output[output_idx];
                                }
                            }
                        }
                    }
                    int filter_idx = out_ch * (in_channels * (2*r+1) * (2*r+1)) + in_ch * ((2*r+1) * (2*r+1)) + m * (2*r+1) + n;
                    grad_filter[filter_idx] = grad_sum;
                }
            }
        }
    }
}


void convtranspose2d_forward(float *input, float *output, float *filter, int stride, int r, int in_channels, int out_channels, int i_height, int i_width, int o_height, int o_width){    
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < out_channels; j++){
            for (int k = 0; k < i_height; k++){
                for (int l = 0; l < i_width; l++){
                    for (int in_ch = 0; in_ch < in_channels; in_ch++){
                        float input_val = input[i * (in_channels * i_height * i_width) + in_ch * (i_height * i_width) + k * i_width + l];
                        for (int m = 0; m < 2*r+1; m++){
                            for (int n = 0; n < 2*r+1; n++){
                                int out_row = k * stride + m - r;
                                int out_col = l * stride + n - r;                                
                                if (out_row >= 0 && out_row < o_height && out_col >= 0 && out_col < o_width){
                                    int filter_idx = j * (in_channels * (2*r+1) * (2*r+1)) + in_ch * ((2*r+1) * (2*r+1)) + m * (2*r+1) + n;
                                    int output_idx = i * (out_channels * o_height * o_width) + j * (o_height * o_width) + out_row * o_width + out_col;                                    
                                    output[output_idx] += filter[filter_idx] * input_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void convtranspose2d_backward(float *input, float *output, float *filter, float *grad_input, float *grad_output, float *grad_filter, int stride, int r, int in_channels, int out_channels, int i_height, int i_width, int o_height, int o_width){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int in_ch = 0; in_ch < in_channels; in_ch++){
            for (int k = 0; k < i_height; k++){
                for (int l = 0; l < i_width; l++){
                    float Pval = 0.0f;
                    for (int j = 0; j < out_channels; j++){
                        for (int m = 0; m < 2*r+1; m++){
                            for (int n = 0; n < 2*r+1; n++){
                                int out_row = k * stride + m - r;
                                int out_col = l * stride + n - r; 
                                if (out_row >= 0 && out_row < o_height && out_col >= 0 && out_col < o_width) {
                                    int filter_idx = j * (in_channels * (2*r+1) * (2*r+1)) + in_ch * ((2*r+1) * (2*r+1)) + m * (2*r+1) + n;
                                    int output_idx = i * (out_channels * o_height * o_width) + j * (o_height * o_width) + out_row * o_width + out_col;                                    
                                    Pval += grad_output[output_idx] * filter[filter_idx];
                                }
                            }
                        }
                    }
                    int input_idx = i * (in_channels * i_height * i_width) + in_ch * (i_height * i_width) + k * i_width + l;
                    grad_input[input_idx] = Pval;
                }
            }
        }
    }

    for (int j = 0; j < out_channels; j++){
        for (int in_ch = 0; in_ch < in_channels; in_ch++){
            for (int m = 0; m < 2*r+1; m++){
                for (int n = 0; n < 2*r+1; n++){
                    float Pval = 0.0f;
                    for (int i = 0; i < BATCH_SIZE; i++){
                        for (int k = 0; k < i_height; k++){  
                            for (int l = 0; l < i_width; l++){
                                int out_row = k * stride + m - r;  
                                int out_col = l * stride + n - r;  
                                if (out_row >= 0 && out_row < o_height && out_col >= 0 && out_col < o_width) {
                                    int input_idx = i * (in_channels * i_height * i_width) + in_ch * (i_height * i_width) + k * i_width + l;
                                    int output_idx = i * (out_channels * o_height * o_width) + j * (o_height * o_width) + out_row * o_width + out_col;
                                    Pval += input[input_idx] * grad_output[output_idx];
                                }
                            }
                        }
                    }
                    int filter_idx = j * (in_channels * (2*r+1) * (2*r+1)) + in_ch * ((2*r+1) * (2*r+1)) + m * (2*r+1) + n;
                    grad_filter[filter_idx] = Pval;
                }
            }
        }
    }
}

void relu_forward(float *input, float *output, int in_channels, int height, int width){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < in_channels; j++){
            for (int k = 0; k < height; k++){
                for (int l = 0; l < width; l++){
                    int input_idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    output[input_idx] = fmaxf(0.0f, input[input_idx]);
                }
            }
        }
    }
}

void relu_backward(float *forward_input, float *grad_output, float *grad_input, int in_channels, int height, int width) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int input_idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    if (forward_input[input_idx] > 0) {
                        grad_input[input_idx] += grad_output[input_idx];
                    } else {
                        grad_input[input_idx] += 0.0f;
                    }
                }
            }
        }
    }
}

void bias_conv_forward(float *input, float *bias, int in_channels, int height, int width){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < in_channels; j++){
            for (int k = 0; k < height; k++){
                for (int l = 0; l < width; l++){
                    int idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    input[idx] += bias[j];
                }
            }
        }
    }
}

void bias_conv_backward(float *grad_output, float *grad_bias, int in_channels, int height, int width){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < in_channels; j++){
            for (int k = 0; k < height; k++){
                for (int l = 0; l < width; l++){
                    int idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    grad_bias[j] += grad_output[idx];
                }
            }
        }
    }
}

void bias_forward(float *input, float *bias, int size){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < size; j++){
            input[i * size + j] += bias[j];
        }
    }
}

void bias_backward(float *grad_bias, float *grad, int batch_size, int size) {
    for (int i = 0; i < size; i++) {
        grad_bias[i] = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad[b * size + i];
        }
    }
}



void groupnorm_forward(float *input, float *mean, float *var, int in_channels, int num_groups, int height, int width){
    int group_size = in_channels / num_groups;
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < num_groups; j++){
            float summ = 0.0f;
            for (int k = 0; k < group_size; k++){
                for (int l = 0; l < height; l++){
                    for (int m = 0; m < width; m++){
                        int channel = j * group_size + k;
                        int idx = i * (in_channels * height * width) + channel * (height * width) + l * width + m;
                        summ += input[idx];
                    }
                }
            }
            mean[i * (num_groups) + j] = summ / (group_size * height * width);
        }
    }

    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < num_groups; j++){
            float summ_sq = 0.0f;
            for (int k = 0; k < group_size; k++){
                for (int l = 0; l < height; l++){
                    for (int m = 0; m < width; m++){
                        int channel = j * group_size + k;
                        int idx = i * (in_channels * height * width) + channel * (height * width) + l * width + m;
                        float diff = input[idx] - mean[i * (num_groups) + j];
                        summ_sq += diff * diff;
                    }
                }
            }
            var[i * (num_groups) + j] = summ_sq / (group_size * height * width);
        }
    }

    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < num_groups; j++){
            for (int k = 0; k < group_size; k++){
                for (int l = 0; l < height; l++){
                    for (int m = 0;m < width; m++){
                        int channel = j * group_size + k;
                        int idx = i * (in_channels * height * width) + channel * (height * width) + l * width + m;
                        float denom = var[i * (num_groups) + j] + 1e-8;
                        input[idx] = (input[idx] - mean[i * (num_groups) + j]) / sqrtf(denom);
                    }
                }
            }
        }
    }
}

void groupnorm_backward(float *input, float *mean, float *var, float *grad_input, float *grad_output, float *grad_mean, float *grad_var, int in_channels, int num_groups, int height, int width){
    int group_size = in_channels / num_groups;  
    
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < num_groups; j++){
            float summ = 0.0f;  
            for (int k = 0; k < group_size; k++){
                for (int l = 0; l < height; l++){
                    for (int m = 0; m < width; m++){
                        int channel = j * group_size + k;
                        int idx = i * (in_channels * height * width) + channel * (height * width) + l * width + m;
                
                        summ += grad_output[idx] * (input[idx] - mean[i * (num_groups) + j]);
                    }
                }
            }
            grad_var[i * (num_groups) + j] = summ * (-0.5f) * powf(var[i * (num_groups) + j] + 1e-8, -1.5f);
        }
    }

    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < num_groups; j++){
            float term_1 = 0.0f;
            float term_2 = 0.0f;
            for (int k = 0; k < group_size; k++){
                for (int l = 0; l < height; l++){
                    for (int m = 0; m < width; m++){
                        int channel = j * group_size + k;
                        int idx = i * (in_channels * height * width) + channel * (height * width) + l * width + m;
                        term_1 += grad_output[idx] * (-1.0f / sqrtf(var[i * (num_groups) + j] + 1e-8));
                        float var_grad_wrt_mean = (-2.0f / (group_size * height * width)) * (input[idx] - mean[i * (num_groups) + j]);
                        term_2 += grad_output[idx] * var_grad_wrt_mean * (-0.5f) * (input[idx] - mean[i * (num_groups) + j]) * powf(var[i * (num_groups) + j] + 1e-8, -1.5f);
                    }
                }
            }
            grad_mean[i * (num_groups) + j] = term_1 + term_2;
        }
    }

    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < num_groups; j++){
            for (int k = 0; k < group_size; k++){
                for (int l = 0; l < height; l++){
                    for (int m = 0; m < width; m++){
                        int channel = j * group_size + k;
                        int idx = i * (in_channels * height * width) + channel * (height * width) + l * width + m;
                        float term_1 = grad_output[idx] * (1.0f / sqrtf(var[i * (num_groups) + j] + 1e-8));
                        float term_2 = grad_var[i * (num_groups) + j] * (2.0f/(group_size * height * width)) * (input[idx] - mean[i * (num_groups) + j]);
                        float term_3 = grad_mean[i * (num_groups) + j] * (1.0f/(group_size * height * width));
                        
                        grad_input[idx] += term_1 + term_2 + term_3;
                    }
                }
            }
        }
    }
}


void maxpool2d_forward(float *input, float *output, int kernel_size, int in_channels, int i_height, int i_width, int o_height, int o_width){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < in_channels; j++){
            for (int k = 0; k < o_height; k++){
                for (int l = 0; l < o_width; l++){
                    float maxx = -INFINITY;
                    for (int m = 0; m < kernel_size; m++){
                        for (int n = 0; n < kernel_size; n++){
                            int input_row = k * kernel_size + m;
                            int input_col = l * kernel_size + n;
                            int input_idx = i * (in_channels * i_height * i_width) + j * (i_height * i_width) + input_row * i_width + input_col;
                            maxx = fmaxf(maxx, input[input_idx]);
                        }
                    }
                    int output_idx = i * (in_channels * o_height * o_width) + j * (o_height * o_width) + k * o_width + l;
                    output[output_idx] = maxx;
                }
            }
        }
    }
}

void maxpool2d_backward(float *input, float *output, float *grad_input, float *grad_output, int kernel_size, int in_channels, int i_height, int i_width, int o_height, int o_width){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < in_channels; j++){
            for (int k = 0; k < o_height; k++){
                for (int l = 0; l < o_width; l++){
                    for (int m = 0; m < kernel_size; m++){
                        for (int n = 0; n < kernel_size; n++){
                            int input_row = k * kernel_size + m;
                            int input_col = l * kernel_size + n;
                            int input_idx = i * (in_channels * i_height * i_width) + j * (i_height * i_width) + input_row * i_width + input_col;
                            int output_idx = i * (in_channels * o_height * o_width) + j * (o_height * o_width) + k * o_width + l;
                            if (output[output_idx] == input[input_idx]){
                                grad_input[input_idx] = grad_output[output_idx];
                            } 
                        }
                    }
                }
            }
        }
    }
}

void sin_forward(float *input, float *output){
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < TIME_HIDDEN_SIZE; j++){
            int idx = i * (TIME_HIDDEN_SIZE) + j;
            output[idx] = sin(input[idx]);
        }
    }
}

void sin_backward(float *input, float *grad_output, float *grad_input) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < TIME_HIDDEN_SIZE; j++) {
            int idx = i * (TIME_HIDDEN_SIZE) + j;
            grad_input[idx] += grad_output[idx] * cos(input[idx]);
        }
    }
}


void ddpm_schedule(float *alpha, float *oneover_sqrta, float *sqrt_beta, float *alphabar, float *sqrtab, float *sqrtmab, float *mab_over_sqrtmab){
    float beta[TIMESTEPS];
    float log_alpha[TIMESTEPS];
    float summ = 0.0f;
    
    for (int i = 0; i < TIMESTEPS; i++){
        beta[i] = (BETA_2 - BETA_1) * ((float)i) / TIMESTEPS + BETA_1;
        sqrt_beta[i] = sqrtf(beta[i]);
        alpha[i] = 1.0f - beta[i];
        log_alpha[i] = logf(alpha[i]);
        summ += log_alpha[i];
        alphabar[i] = expf(summ);
        sqrtab[i] = sqrtf(alphabar[i]);
        oneover_sqrta[i] = 1.0f / sqrtf(alpha[i]);
        sqrtmab[i] = sqrtf(1.0f - alphabar[i]);
        mab_over_sqrtmab[i] = (1.0f - alpha[i]) / sqrtmab[i];
    }
}

void noise_addition(float *input, float *output, float *noise, int *timesteps, float *sqrt_alpha_bar, float *sqrt_one_minus_alpha_bar, int batch_size, int in_channels, int height, int width) {
    for (int i = 0; i < batch_size; i++) {
        int t = timesteps[i];
        float sqrt_alpha_bar1 = sqrt_alpha_bar[t];
        float sqrt_one_minus_alpha_bar1 = sqrt_one_minus_alpha_bar[t];
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;        
                    output[idx] = (sqrt_alpha_bar1 * input[idx]) + (sqrt_one_minus_alpha_bar1 * noise[idx]);
                }
            }
        }
    }
}

void noise_addition_backward(float *grad_output, float *grad_input, float *grad_noise, int *timesteps, float *sqrt_alpha_bar, float *sqrt_one_minus_alpha_bar, int batch_size, int in_channels, int height, int width) {
    for (int i = 0; i < batch_size; i++) {
        int t = timesteps[i];
        float sqrt_alpha_bar1 = sqrt_alpha_bar[t];
        float sqrt_one_minus_alpha_bar1 = sqrt_one_minus_alpha_bar[t];
        
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    grad_input[idx] += grad_output[idx] * sqrt_alpha_bar1;
                    grad_noise[idx] += grad_output[idx] * sqrt_one_minus_alpha_bar1;
                }
            }
        }
    }
}

void randn(float *output, int size, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 12; j++) {
            sum += (float)rand() / (float)RAND_MAX;
        }
        output[i] = sum - 6.0f;
    }
}

void randint_with_seed(float *output, int low, int high, int size, unsigned int seed) {
    srand(seed);
    int range = high - low;
    for (int i = 0; i < size; i++) {
        output[i] = low + (rand() % range);
    }
}

void add_time_embedding(float *input, float *time_input, float *output, int batch_size, int in_channels, int height, int width) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    output[idx] = input[idx] + time_input[i * in_channels + j];
                }
            }
        }
    }
}

void add_time_embedding_backward(float *grad_output, float *grad_input, float *grad_time_input, int batch_size, int in_channels, int height, int width) {    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    grad_input[idx] += grad_output[idx];
                    grad_time_input[i * in_channels + j] += grad_output[idx];
                }
            }
        }
    }
}

float mse_loss(float *output, float *labels, int in_channels, int height, int width){
    float loss = 0.0f;
    for (int i = 0; i < BATCH_SIZE; i++){
        for (int j = 0; j < in_channels; j++){
            for (int k = 0; k < height; k++){
                for (int l = 0; l < width; l++){
                    int idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    float diff = output[idx] - labels[idx];
                    loss += (diff * diff);
                }
            }
        }
    }

    return loss / (BATCH_SIZE * in_channels * height * width);
}

void mse_loss_backward(float *output, float *labels, float *grad_output, int in_channels, int height, int width) {
    float N = BATCH_SIZE * in_channels * height * width;
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int idx = i * (in_channels * height * width) + j * (height * width) + k * width + l;
                    grad_output[idx] = (2.0f / N) * (output[idx] - labels[idx]); 
                }
            }
        }
    }
}

void zero_grad(float *grad, int size) {
    memset(grad, 0, size * sizeof(float));
}

void update_weights_timed(Network *nn) {    
    
    
    // Update filter weights
    for (int i = 0; i < CHANNEL_1 * 1 * 3 * 3; i++) {
        nn->filter_1[i] -= LEARNING_RATE * nn->grad_filter_1[i];
    }
    for (int i = 0; i < CHANNEL_2 * CHANNEL_1 * 3 * 3; i++) {
        nn->filter_2[i] -= LEARNING_RATE * nn->grad_filter_2[i];
    }
    for (int i = 0; i < CHANNEL_3 * CHANNEL_2 * 3 * 3; i++) {
        nn->filter_3[i] -= LEARNING_RATE * nn->grad_filter_3[i];
    }
    for (int i = 0; i < CHANNEL_2 * CHANNEL_3 * 3 * 3; i++) {
        nn->filter_4[i] -= LEARNING_RATE * nn->grad_filter_4[i];
    }
    for (int i = 0; i < CHANNEL_2 * CHANNEL_2 * 3 * 3; i++) {
        nn->filter_41[i] -= LEARNING_RATE * nn->grad_filter_41[i];
    }
    for (int i = 0; i < CHANNEL_1 * CHANNEL_2 * 3 * 3; i++) {
        nn->filter_5[i] -= LEARNING_RATE * nn->grad_filter_5[i];
    }
    for (int i = 0; i < CHANNEL_1 * CHANNEL_1 * 3 * 3; i++) {
        nn->filter_51[i] -= LEARNING_RATE * nn->grad_filter_51[i];
    }
    for (int i = 0; i < 1 * CHANNEL_1 * 1 * 1; i++) {
        nn->filter_6[i] -= LEARNING_RATE * nn->grad_filter_6[i];
    }
    
    // Update bias weights
    for (int i = 0; i < CHANNEL_1; i++) {
        nn->bias_1[i] -= LEARNING_RATE * nn->grad_bias_1[i];
    }
    for (int i = 0; i < CHANNEL_2; i++) {
        nn->bias_2[i] -= LEARNING_RATE * nn->grad_bias_2[i];
    }
    for (int i = 0; i < CHANNEL_3; i++) {
        nn->bias_3[i] -= LEARNING_RATE * nn->grad_bias_3[i];
    }
    for (int i = 0; i < CHANNEL_2; i++) {
        nn->bias_4[i] -= LEARNING_RATE * nn->grad_bias_4[i];
    }
    for (int i = 0; i < CHANNEL_2; i++) {
        nn->bias_41[i] -= LEARNING_RATE * nn->grad_bias_41[i];
    }
    for (int i = 0; i < CHANNEL_1; i++) {
        nn->bias_5[i] -= LEARNING_RATE * nn->grad_bias_5[i];
    }
    for (int i = 0; i < CHANNEL_1; i++) {
        nn->bias_51[i] -= LEARNING_RATE * nn->grad_bias_51[i];
    }
    for (int i = 0; i < 1; i++) {
        nn->bias_6[i] -= LEARNING_RATE * nn->grad_bias_6[i];
    }
    
    // Update matmul weights
    for (int i = 0; i < 1 * TIME_HIDDEN_SIZE; i++) {
        nn->matmul_1[i] -= LEARNING_RATE * nn->grad_matmul_1[i];
    }
    for (int i = 0; i < TIME_HIDDEN_SIZE * TIME_HIDDEN_SIZE; i++) {
        nn->matmul_2[i] -= LEARNING_RATE * nn->grad_matmul_2[i];
    }
    for (int i = 0; i < TIME_HIDDEN_SIZE; i++) {
        nn->bias_m2[i] -= LEARNING_RATE * nn->grad_bias_m2[i];
    }
    
}

void forward_timed(Network *nn, float *batch_input, float *time, float *output, 
                   float *conv_out1, float *conv_outbr1, float *conv_outbr_max1,
                   float *conv_out2, float *conv_outbr2, float *conv_outbr_max2, 
                   float *conv_out3, float *conv_outbr3, float *conv_out4,
                   float *conv_outbc4, float *conv_outbr4, float *conv_out5, 
                   float *conv_outbc5, float *conv_outbr5,
                   float *mean_1, float *std_1, float *mean_2, float *std_2, 
                   float *mean_3, float *std_3, float *mean_4, float *std_4, 
                   float *mean_5, float *std_5, float *matmul_out1, float *matmul_outs1, 
                   float *matmul_out2, float *conv_out_concat) {
    
    // enc block1
    conv2d_forward(batch_input, conv_out1, nn->filter_1, 1, 1, 16, 28, 28);
    bias_conv_forward(conv_out1, nn->bias_1, 16, 28, 28);
    groupnorm_forward(conv_out1, mean_1, std_1, 16, 4, 28, 28);
    relu_forward(conv_out1, conv_outbr1, 16, 28, 28);
    maxpool2d_forward(conv_outbr1, conv_outbr_max1, 2, 16, 28, 28, 14, 14);
    
    // enc block2
    conv2d_forward(conv_outbr_max1, conv_out2, nn->filter_2, 1, 16, 32, 14, 14);
    bias_conv_forward(conv_out2, nn->bias_2, 32, 14, 14);
    groupnorm_forward(conv_out2, mean_2, std_2, 32, 4, 14, 14);
    relu_forward(conv_out2, conv_outbr2, 32, 14, 14);
    maxpool2d_forward(conv_outbr2, conv_outbr_max2, 2, 32, 14, 14, 7, 7);
    
    // bottleneck
    conv2d_forward(conv_outbr_max2, conv_out3, nn->filter_3, 1, 32, 64, 7, 7);
    groupnorm_forward(conv_out3, mean_3, std_3, 64, 8, 7, 7);
    relu_forward(conv_out3, conv_outbr3, 64, 7, 7);
    
    // time embeddings
    matmul_a_b(time, nn->matmul_1, matmul_out1, BATCH_SIZE, 1, TIME_HIDDEN_SIZE);
    sin_forward(matmul_out1, matmul_outs1);
    matmul_a_b(matmul_outs1, nn->matmul_2, matmul_out2, BATCH_SIZE, TIME_HIDDEN_SIZE, TIME_HIDDEN_SIZE);
    bias_forward(matmul_out2, nn->bias_m2, TIME_HIDDEN_SIZE);
    
    // concatenate output + timeembeddings
    add_time_embedding(conv_outbr3, matmul_out2, conv_out_concat, BATCH_SIZE, 64, 7, 7);
    
    // dec block1
    convtranspose2d_forward(conv_out_concat, conv_out4, nn->filter_4, 2, 1, 64, 32, 7, 7, 14, 14);
    bias_conv_forward(conv_out4, nn->bias_4, 32, 14, 14);
    conv2d_forward(conv_out4, conv_outbc4, nn->filter_41, 1, 32, 32, 14, 14);
    bias_conv_forward(conv_outbc4, nn->bias_41, 32, 14, 14);
    groupnorm_forward(conv_outbc4, mean_4, std_4, 32, 4, 14, 14);
    relu_forward(conv_outbc4, conv_outbr4, 32, 14, 14);
    
    // dec block2
    convtranspose2d_forward(conv_outbr4, conv_out5, nn->filter_5, 2, 1, 32, 16, 14, 14, 28, 28);
    bias_conv_forward(conv_out5, nn->bias_5, 16, 28, 28);
    conv2d_forward(conv_out5, conv_outbc5, nn->filter_51, 1, 16, 16, 28, 28);
    bias_conv_forward(conv_outbc5, nn->bias_51, 16, 28, 28);
    groupnorm_forward(conv_outbc5, mean_5, std_5, 16, 4, 28, 28);
    relu_forward(conv_outbc5, conv_outbr5, 16, 28, 28);
    
    // output layer
    conv2d_forward(conv_outbr5, output, nn->filter_6, 1, 16, 1, 28, 28);
    bias_conv_forward(output, nn->bias_6, 1, 28, 28);
}

void backward_timed(Network *nn, float *batch_input, float *time, float *output, 
                   float *conv_out1, float *conv_outbr1, float *conv_outbr_max1, 
                   float *conv_out2, float *conv_outbr2, float *conv_outbr_max2, 
                   float *conv_out3, float *conv_outbr3, float *conv_out4, 
                   float *conv_outbc4, float *conv_outbr4, float *conv_out5, 
                   float *conv_outbc5, float *conv_outbr5, 
                   float *mean_1, float *std_1, float *mean_2, float *std_2, 
                   float *mean_3, float *std_3, float *mean_4, float *std_4, 
                   float *mean_5, float *std_5, float *matmul_out1, float *matmul_outs1, 
                   float *matmul_out2, float *conv_out_concat, float *true_noise) {
    
    // Initialize all gradient arrays to zero
    zero_grad(nn->grad_filter_1, CHANNEL_1 * 1 * 3 * 3);
    zero_grad(nn->grad_filter_2, CHANNEL_2 * CHANNEL_1 * 3 * 3);
    zero_grad(nn->grad_filter_3, CHANNEL_3 * CHANNEL_2 * 3 * 3);
    zero_grad(nn->grad_filter_4, CHANNEL_2 * CHANNEL_3 * 3 * 3);
    zero_grad(nn->grad_filter_41, CHANNEL_2 * CHANNEL_2 * 3 * 3);
    zero_grad(nn->grad_filter_5, CHANNEL_1 * CHANNEL_2 * 3 * 3);
    zero_grad(nn->grad_filter_51, CHANNEL_1 * CHANNEL_1 * 3 * 3);
    zero_grad(nn->grad_filter_6, 1 * CHANNEL_1 * 1 * 1);
    
    // Initialize bias gradients to zero
    zero_grad(nn->grad_bias_1, CHANNEL_1);
    zero_grad(nn->grad_bias_2, CHANNEL_2);
    zero_grad(nn->grad_bias_3, CHANNEL_3);
    zero_grad(nn->grad_bias_4, CHANNEL_2);
    zero_grad(nn->grad_bias_41, CHANNEL_2);
    zero_grad(nn->grad_bias_5, CHANNEL_1);
    zero_grad(nn->grad_bias_51, CHANNEL_1);
    zero_grad(nn->grad_bias_6, 1);
    
    // Initialize matmul gradients to zero
    zero_grad(nn->grad_matmul_1, 1 * TIME_HIDDEN_SIZE);
    zero_grad(nn->grad_matmul_2, TIME_HIDDEN_SIZE * TIME_HIDDEN_SIZE);
    zero_grad(nn->grad_bias_m2, TIME_HIDDEN_SIZE);
    
    // Allocate memory for gradients with proper sizes
    float *grad_output_final = malloc(BATCH_SIZE * 1 * 28 * 28 * sizeof(float));
    mse_loss_backward(output, true_noise, grad_output_final, 1, 28, 28);
    
    // output layer
    float *grad_conv_outbr5 = malloc(BATCH_SIZE * 16 * 28 * 28 * sizeof(float));
    bias_conv_backward(grad_output_final, nn->grad_bias_6, 1, 28, 28);
    conv2d_backward(conv_outbr5, output, nn->filter_6, grad_conv_outbr5, grad_output_final, nn->grad_filter_6, 1, 16, 1, 28, 28);

    // dec block2
    float *grad_conv_outbc5 = malloc(BATCH_SIZE * 16 * 28 * 28 * sizeof(float));
    relu_backward(conv_outbc5, grad_conv_outbr5, grad_conv_outbc5, 16, 28, 28);
    float *grad_mean_5 = malloc(BATCH_SIZE * 4 * sizeof(float));
    float *grad_var_5 = malloc(BATCH_SIZE * 4 * sizeof(float));
    groupnorm_backward(conv_outbc5, mean_5, std_5, grad_conv_outbc5, grad_conv_outbc5, grad_mean_5, grad_var_5, 16, 4, 28, 28);
    bias_conv_backward(grad_conv_outbc5, nn->grad_bias_51, 16, 28, 28);
    float *grad_conv_out5 = malloc(BATCH_SIZE * 16 * 28 * 28 * sizeof(float));
    conv2d_backward(conv_out5, conv_outbc5, nn->filter_51, grad_conv_out5, grad_conv_outbc5, nn->grad_filter_51, 1, 16, 16, 28, 28);
    bias_conv_backward(grad_conv_out5, nn->grad_bias_5, 16, 28, 28);
    float *grad_conv_outbr4 = malloc(BATCH_SIZE * 32 * 14 * 14 * sizeof(float));
    convtranspose2d_backward(conv_outbr4, conv_out5, nn->filter_5, grad_conv_outbr4, grad_conv_out5, nn->grad_filter_5, 2, 1, 32, 16, 14, 14, 28, 28);

    // dec block1
    float *grad_conv_outbc4 = malloc(BATCH_SIZE * 32 * 14 * 14 * sizeof(float));
    relu_backward(conv_outbc4, grad_conv_outbr4, grad_conv_outbc4, 32, 14, 14);
    float *grad_mean_4 = malloc(BATCH_SIZE * 8 * sizeof(float));
    float *grad_var_4 = malloc(BATCH_SIZE * 8 * sizeof(float));
    groupnorm_backward(conv_outbc4, mean_4, std_4, grad_conv_outbc4, grad_conv_outbc4, grad_mean_4, grad_var_4, 32, 4, 14, 14);
    bias_conv_backward(grad_conv_outbc4, nn->grad_bias_41, 32, 14, 14);
    float *grad_conv_out4 = malloc(BATCH_SIZE * 32 * 14 * 14 * sizeof(float));
    conv2d_backward(conv_out4, conv_outbc4, nn->filter_41, grad_conv_out4, grad_conv_outbc4, nn->grad_filter_41, 1, 32, 32, 14, 14);
    bias_conv_backward(grad_conv_out4, nn->grad_bias_4, 32, 14, 14);
    float *grad_conv_out_concat = malloc(BATCH_SIZE * 64 * 7 * 7 * sizeof(float));
    convtranspose2d_backward(conv_out_concat, conv_out4, nn->filter_4, grad_conv_out_concat, grad_conv_out4, nn->grad_filter_4, 2, 1, 64, 32, 7, 7, 14, 14);
    
    // concatenate output + timeembeddings
    float *grad_conv_outbr3 = malloc(BATCH_SIZE * 64 * 7 * 7 * sizeof(float));
    float *grad_matmul_out2 = malloc(BATCH_SIZE * TIME_HIDDEN_SIZE * sizeof(float));
    add_time_embedding_backward(grad_conv_out_concat, grad_conv_outbr3, grad_matmul_out2, BATCH_SIZE, 64, 7, 7);
    
    // time embeddings
    bias_backward(grad_matmul_out2, nn->grad_bias_m2, BATCH_SIZE, TIME_HIDDEN_SIZE);
    float *grad_matmul_outs1 = malloc(BATCH_SIZE * TIME_HIDDEN_SIZE * sizeof(float));
    matmul_a_bt(grad_matmul_out2, nn->matmul_2, grad_matmul_outs1, BATCH_SIZE, TIME_HIDDEN_SIZE, TIME_HIDDEN_SIZE);
    matmul_at_b(matmul_outs1, grad_matmul_out2, nn->grad_matmul_2, BATCH_SIZE, TIME_HIDDEN_SIZE, TIME_HIDDEN_SIZE);
    float *grad_matmul_out1 = malloc(BATCH_SIZE * TIME_HIDDEN_SIZE * sizeof(float));
    sin_backward(matmul_out1, grad_matmul_outs1, grad_matmul_out1);
    float *grad_time = malloc(BATCH_SIZE * 1 * sizeof(float));
    matmul_a_bt(grad_matmul_out1, nn->matmul_1, grad_time, BATCH_SIZE, 1, TIME_HIDDEN_SIZE);
    matmul_at_b(time, grad_matmul_out1, nn->grad_matmul_1, BATCH_SIZE, 1, TIME_HIDDEN_SIZE);

    // bottleneck
    float *grad_conv_out3 = malloc(BATCH_SIZE * 64 * 7 * 7 * sizeof(float));
    relu_backward(conv_out3, grad_conv_outbr3, grad_conv_out3, 64, 7, 7);
    float *grad_mean_3 = malloc(BATCH_SIZE * 8 * sizeof(float));
    float *grad_var_3 = malloc(BATCH_SIZE * 8 * sizeof(float));
    groupnorm_backward(conv_out3, mean_3, std_3, grad_conv_out3, grad_conv_out3, grad_mean_3, grad_var_3, 64, 8, 7, 7);
    float *grad_conv_outbr_max2 = malloc(BATCH_SIZE * 32 * 7 * 7 * sizeof(float));
    conv2d_backward(conv_outbr_max2, conv_out3, nn->filter_3, grad_conv_outbr_max2, grad_conv_out3, nn->grad_filter_3, 1, 32, 64, 7, 7);

    // enc block2    
    float *grad_conv_outbr2 = malloc(BATCH_SIZE * 32 * 14 * 14 * sizeof(float));
    maxpool2d_backward(conv_outbr2, conv_outbr_max2, grad_conv_outbr2, grad_conv_outbr_max2, 2, 32, 14, 14, 7, 7);
    float *grad_conv_out2 = malloc(BATCH_SIZE * 32 * 14 * 14 * sizeof(float));
    relu_backward(conv_out2, grad_conv_outbr2, grad_conv_out2, 32, 14, 14);
    float *grad_mean_2 = malloc(BATCH_SIZE * 8 * sizeof(float));
    float *grad_var_2 = malloc(BATCH_SIZE * 8 * sizeof(float));
    groupnorm_backward(conv_out2, mean_2, std_2, grad_conv_out2, grad_conv_out2, grad_mean_2, grad_var_2, 32, 4, 14, 14);
    bias_conv_backward(grad_conv_out2, nn->grad_bias_2, 32, 14, 14);
    float *grad_conv_outbr_max1 = malloc(BATCH_SIZE * 16 * 14 * 14 * sizeof(float));
    conv2d_backward(conv_outbr_max1, conv_out2, nn->filter_2, grad_conv_outbr_max1, grad_conv_out2, nn->grad_filter_2, 1, 16, 32, 14, 14);

    // enc block1
    float *grad_conv_outbr1 = malloc(BATCH_SIZE * 16 * 28 * 28 * sizeof(float));
    maxpool2d_backward(conv_outbr1, conv_outbr_max1, grad_conv_outbr1, grad_conv_outbr_max1, 2, 16, 28, 28, 14, 14);
    float *grad_conv_out1 = malloc(BATCH_SIZE * 16 * 28 * 28 * sizeof(float));
    relu_backward(conv_out1, grad_conv_outbr1, grad_conv_out1, 16, 28, 28);
    float *grad_mean_1 = malloc(BATCH_SIZE * 4 * sizeof(float));
    float *grad_var_1 = malloc(BATCH_SIZE * 4 * sizeof(float));
    groupnorm_backward(conv_out1, mean_1, std_1, grad_conv_out1, grad_conv_out1, grad_mean_1, grad_var_1, 16, 4, 28, 28);
    bias_conv_backward(grad_conv_out1, nn->grad_bias_1, 16, 28, 28);
    float *grad_batch_input = malloc(BATCH_SIZE * 1 * 28 * 28 * sizeof(float));
    conv2d_backward(batch_input, conv_out1, nn->filter_1, grad_batch_input, grad_conv_out1, nn->grad_filter_1, 1, 1, 16, 28, 28);

    // Free allocated memory
    free(grad_output_final);
    free(grad_conv_outbr5);
    free(grad_conv_outbc5);
    free(grad_mean_5);
    free(grad_var_5);
    free(grad_conv_out5);
    free(grad_conv_outbr4);
    free(grad_conv_outbc4);
    free(grad_mean_4);
    free(grad_var_4);
    free(grad_conv_out4);
    free(grad_conv_out_concat);
    free(grad_conv_outbr3);
    free(grad_matmul_out2);
    free(grad_matmul_outs1);
    free(grad_matmul_out1);
    free(grad_time);
    free(grad_conv_out3);
    free(grad_mean_3);
    free(grad_var_3);
    free(grad_conv_outbr_max2);
    free(grad_conv_outbr2);
    free(grad_conv_out2);
    free(grad_mean_2);
    free(grad_var_2);
    free(grad_conv_outbr_max1);
    free(grad_conv_outbr1);
    free(grad_conv_out1);
    free(grad_mean_1);
    free(grad_var_1);
    free(grad_batch_input);
}


void train_timed(Network *nn, float *X_train) {    
    // Encoder layer 1
    float *conv_out1 = malloc(BATCH_SIZE * CHANNEL_1 * HEIGHT * WIDTH * sizeof(float));
    float *conv_outbr1 = malloc(BATCH_SIZE * CHANNEL_1 * HEIGHT * WIDTH * sizeof(float));
    float *conv_outbr_max1 = malloc(BATCH_SIZE * CHANNEL_1 * (HEIGHT/2) * (WIDTH/2) * sizeof(float));

    // Encoder layer 2  
    float *conv_out2 = malloc(BATCH_SIZE * CHANNEL_2 * (HEIGHT/2) * (WIDTH/2) * sizeof(float));
    float *conv_outbr2 = malloc(BATCH_SIZE * CHANNEL_2 * (HEIGHT/2) * (WIDTH/2) * sizeof(float));
    float *conv_outbr_max2 = malloc(BATCH_SIZE * CHANNEL_2 * (HEIGHT/4) * (WIDTH/4) * sizeof(float));

    // Bottleneck
    float *conv_out3 = malloc(BATCH_SIZE * CHANNEL_3 * (HEIGHT/4) * (WIDTH/4) * sizeof(float));
    float *conv_outbr3 = malloc(BATCH_SIZE * CHANNEL_3 * (HEIGHT/4) * (WIDTH/4) * sizeof(float));

    // Time embeddings
    float *matmul_out1 = malloc(BATCH_SIZE * TIME_HIDDEN_SIZE * sizeof(float));
    float *matmul_outs1 = malloc(BATCH_SIZE * TIME_HIDDEN_SIZE * sizeof(float));
    float *matmul_out2 = malloc(BATCH_SIZE * TIME_HIDDEN_SIZE * sizeof(float));

    // Add time embeddings
    float *conv_out_concat = malloc(BATCH_SIZE * CHANNEL_3 * (HEIGHT/4) * (WIDTH/4) * sizeof(float));

    // Decoder layer 1
    float *conv_out4 = malloc(BATCH_SIZE * CHANNEL_2 * (HEIGHT/2) * (WIDTH/2) * sizeof(float));
    float *conv_outbc4 = malloc(BATCH_SIZE * CHANNEL_2 * (HEIGHT/2) * (WIDTH/2) * sizeof(float));
    float *conv_outbr4 = malloc(BATCH_SIZE * CHANNEL_2 * (HEIGHT/2) * (WIDTH/2) * sizeof(float));

    // Decoder layer 2
    float *conv_out5 = malloc(BATCH_SIZE * CHANNEL_1 * HEIGHT * WIDTH * sizeof(float));
    float *conv_outbc5 = malloc(BATCH_SIZE * CHANNEL_1 * HEIGHT * WIDTH * sizeof(float));
    float *conv_outbr5 = malloc(BATCH_SIZE * CHANNEL_1 * HEIGHT * WIDTH * sizeof(float));

    // GroupNorm statistics
    float *mean_1 = malloc(BATCH_SIZE * 4 * sizeof(float));
    float *std_1 = malloc(BATCH_SIZE * 4 * sizeof(float));
    float *mean_2 = malloc(BATCH_SIZE * 8 * sizeof(float));
    float *std_2 = malloc(BATCH_SIZE * 8 * sizeof(float));      
    float *mean_3 = malloc(BATCH_SIZE * 8 * sizeof(float));
    float *std_3 = malloc(BATCH_SIZE * 8 * sizeof(float));       
    float *mean_4 = malloc(BATCH_SIZE * 8 * sizeof(float));
    float *std_4 = malloc(BATCH_SIZE * 8 * sizeof(float));      
    float *mean_5 = malloc(BATCH_SIZE * 4 * sizeof(float));
    float *std_5 = malloc(BATCH_SIZE * 4 * sizeof(float));  
    
    float *output = malloc(BATCH_SIZE * 1 * HEIGHT * WIDTH * sizeof(float));

    // DDPM schedule parameters
    float *alpha = malloc(TIMESTEPS * sizeof(float));
    float *oneover_sqrta = malloc(TIMESTEPS * sizeof(float));
    float *sqrt_beta = malloc(TIMESTEPS * sizeof(float));
    float *alphabar = malloc(TIMESTEPS * sizeof(float));
    float *sqrtab = malloc(TIMESTEPS * sizeof(float));
    float *sqrtmab = malloc(TIMESTEPS * sizeof(float));
    float *mab_over_sqrtmab = malloc(TIMESTEPS * sizeof(float));

    // Noise and timesteps
    float *noise_added_input = malloc(BATCH_SIZE * 1 * HEIGHT * WIDTH * sizeof(float));
    float *true_noise = malloc(BATCH_SIZE * 1 * HEIGHT * WIDTH * sizeof(float));
    float *time = malloc(BATCH_SIZE * 1 * sizeof(float));
    int *timesteps = malloc(BATCH_SIZE * sizeof(int));

    ddpm_schedule(alpha, oneover_sqrta, sqrt_beta, alphabar, sqrtab, sqrtmab, mab_over_sqrtmab);

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            float *batch_input = &X_train[start_idx * HEIGHT * WIDTH];

            // Generate random timesteps for this batch
            for (int i = 0; i < BATCH_SIZE; i++) {
                timesteps[i] = rand() % TIMESTEPS;
                time[i] = (float)timesteps[i] / (float)TIMESTEPS;
            }

            randn(true_noise, BATCH_SIZE * HEIGHT * WIDTH, 42);

            // Add noise to input based on timesteps using DDPM schedule
            noise_addition(batch_input, noise_added_input, true_noise, timesteps, sqrtab, sqrtmab, BATCH_SIZE, 1, HEIGHT, WIDTH);

            // Forward pass
            forward_timed(nn, noise_added_input, time, output, 
                         conv_out1, conv_outbr1, conv_outbr_max1, 
                         conv_out2, conv_outbr2, conv_outbr_max2, 
                         conv_out3, conv_outbr3, conv_out4, 
                         conv_outbc4, conv_outbr4, conv_out5, 
                         conv_outbc5, conv_outbr5, 
                         mean_1, std_1, mean_2, std_2, 
                         mean_3, std_3, mean_4, std_4, 
                         mean_5, std_5, matmul_out1, matmul_outs1, 
                         matmul_out2, conv_out_concat);

            // Calculate loss (predicting the noise)
            float loss = mse_loss(output, true_noise, 1, HEIGHT, WIDTH);
            total_loss += loss;

            // Backward pass
            backward_timed(nn, noise_added_input, time, output, 
                          conv_out1, conv_outbr1, conv_outbr_max1, 
                          conv_out2, conv_outbr2, conv_outbr_max2, 
                          conv_out3, conv_outbr3, conv_out4, 
                          conv_outbc4, conv_outbr4, conv_out5, 
                          conv_outbc5, conv_outbr5, 
                          mean_1, std_1, mean_2, std_2, 
                          mean_3, std_3, mean_4, std_4, 
                          mean_5, std_5, matmul_out1, matmul_outs1, 
                          matmul_out2, conv_out_concat, true_noise);
            
            // Weight update with timing
            update_weights_timed(nn);
        }
        printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
    }

    // Free all allocated memory
    free(conv_out1); free(conv_outbr1); free(conv_outbr_max1);
    free(conv_out2); free(conv_outbr2); free(conv_outbr_max2);
    free(conv_out3); free(conv_outbr3);
    free(matmul_out1); free(matmul_outs1); free(matmul_out2);
    free(conv_out_concat);
    free(conv_out4); free(conv_outbc4); free(conv_outbr4);
    free(conv_out5); free(conv_outbc5); free(conv_outbr5);
    free(mean_1); free(std_1); free(mean_2); free(std_2);
    free(mean_3); free(std_3); free(mean_4); free(std_4);
    free(mean_5); free(std_5);
    free(output);
    free(alpha); free(oneover_sqrta); free(sqrt_beta);
    free(alphabar); free(sqrtab); free(sqrtmab); free(mab_over_sqrtmab);
    free(noise_added_input); free(true_noise); free(time); free(timesteps);
}

void initialize_random_weights(Network *nn) {
    initialize_weights(nn->filter_1, CHANNEL_1, 1, 3, 3);
    initialize_weights(nn->filter_2, CHANNEL_2, CHANNEL_1, 3, 3);
    initialize_weights(nn->filter_3, CHANNEL_3, CHANNEL_2, 3, 3);
    initialize_weights(nn->filter_4, CHANNEL_2, CHANNEL_3, 3, 3);
    initialize_weights(nn->filter_41, CHANNEL_2, CHANNEL_2, 3, 3);
    initialize_weights(nn->filter_5, CHANNEL_1, CHANNEL_2, 3, 3);
    initialize_weights(nn->filter_51, CHANNEL_1, CHANNEL_1, 3, 3);
    initialize_weights(nn->filter_6, 1, CHANNEL_1, 1, 1);

    initialize_bias(nn->bias_1, CHANNEL_1);     // Fixed naming
    initialize_bias(nn->bias_2, CHANNEL_2);
    initialize_bias(nn->bias_3, CHANNEL_3);
    initialize_bias(nn->bias_4, CHANNEL_2);
    initialize_bias(nn->bias_41, CHANNEL_2);
    initialize_bias(nn->bias_5, CHANNEL_1);
    initialize_bias(nn->bias_51, CHANNEL_1);
    initialize_bias(nn->bias_6, 1);

    initialize_matmul_weights(nn->matmul_1, 1, TIME_HIDDEN_SIZE);
    initialize_matmul_weights(nn->matmul_2, TIME_HIDDEN_SIZE, TIME_HIDDEN_SIZE);
    initialize_bias(nn->bias_m2, TIME_HIDDEN_SIZE);  // Fixed function call
}

void initialize_neural_network(Network *nn) {
    // Allocate weights
    nn->filter_1 = malloc(CHANNEL_1 * 1 * 3 * 3 * sizeof(float));  // Fixed
    nn->filter_2 = malloc(CHANNEL_2 * CHANNEL_1 * 3 * 3 * sizeof(float));
    nn->filter_3 = malloc(CHANNEL_3 * CHANNEL_2 * 3 * 3 * sizeof(float));
    nn->filter_4 = malloc(CHANNEL_2 * CHANNEL_3 * 3 * 3 * sizeof(float));
    nn->filter_41 = malloc(CHANNEL_2 * CHANNEL_2 * 3 * 3 * sizeof(float));
    nn->filter_5 = malloc(CHANNEL_1 * CHANNEL_2 * 3 * 3 * sizeof(float));
    nn->filter_51 = malloc(CHANNEL_1 * CHANNEL_1 * 3 * 3 * sizeof(float));
    nn->filter_6 = malloc(1 * CHANNEL_1 * 1 * 1 * sizeof(float));  // Fixed

    // Allocate biases
    nn->bias_1 = malloc(CHANNEL_1 * sizeof(float));
    nn->bias_2 = malloc(CHANNEL_2 * sizeof(float));
    nn->bias_3 = malloc(CHANNEL_3 * sizeof(float));
    nn->bias_4 = malloc(CHANNEL_2 * sizeof(float));
    nn->bias_41 = malloc(CHANNEL_2 * sizeof(float));
    nn->bias_5 = malloc(CHANNEL_1 * sizeof(float));
    nn->bias_51 = malloc(CHANNEL_1 * sizeof(float));
    nn->bias_6 = malloc(1 * sizeof(float));

    // Allocate matmul weights
    nn->matmul_1 = malloc(1 * TIME_HIDDEN_SIZE * sizeof(float));  // Fixed
    nn->matmul_2 = malloc(TIME_HIDDEN_SIZE * TIME_HIDDEN_SIZE * sizeof(float));
    nn->bias_m2 = malloc(TIME_HIDDEN_SIZE * sizeof(float));

    // Allocate gradient arrays (you're missing these!)
    nn->grad_filter_1 = malloc(CHANNEL_1 * 1 * 3 * 3 * sizeof(float));
    // ... allocate all other gradient arrays similarly

    initialize_random_weights(nn);
}

int main() {
    srand(42);

    Network nn;
    initialize_neural_network(&nn);

    // Allocate training data (diffusion models don't need labels)
    float *X_train = malloc(TRAIN_SIZE * HEIGHT * WIDTH * sizeof(float));
    float *X_test = malloc(TEST_SIZE * HEIGHT * WIDTH * sizeof(float));

    // Load and normalize data
    load_data("./data/X_train.bin", X_train, TRAIN_SIZE * HEIGHT * WIDTH);
    normalize_data(X_train, TRAIN_SIZE * HEIGHT * WIDTH);
    load_data("./data/X_test.bin", X_test, TEST_SIZE * HEIGHT * WIDTH);
    normalize_data(X_test, TEST_SIZE * HEIGHT * WIDTH);

    // Train the diffusion model
    train_timed(&nn, X_train);

    // Free network weights
    free(nn.filter_1);
    free(nn.filter_2);
    free(nn.filter_3);
    free(nn.filter_4);
    free(nn.filter_41);
    free(nn.filter_5);
    free(nn.filter_51);
    free(nn.filter_6);

    // Free biases (fixed naming)
    free(nn.bias_1);
    free(nn.bias_2);
    free(nn.bias_3);
    free(nn.bias_4);
    free(nn.bias_41);
    free(nn.bias_5);
    free(nn.bias_51);
    free(nn.bias_6);

    // Free matmul weights
    free(nn.matmul_1);
    free(nn.matmul_2);
    free(nn.bias_m2);

    // Free gradient arrays
    free(nn.grad_filter_1);
    free(nn.grad_filter_2);
    free(nn.grad_filter_3);
    free(nn.grad_filter_4);
    free(nn.grad_filter_41);
    free(nn.grad_filter_5);
    free(nn.grad_filter_51);
    free(nn.grad_filter_6);

    // Free gradient biases (fixed naming)
    free(nn.grad_bias_1);
    free(nn.grad_bias_2);
    free(nn.grad_bias_3);
    free(nn.grad_bias_4);
    free(nn.grad_bias_41);
    free(nn.grad_bias_5);
    free(nn.grad_bias_51);
    free(nn.grad_bias_6);

    // Free gradient matmul arrays
    free(nn.grad_matmul_1);
    free(nn.grad_matmul_2);
    free(nn.grad_bias_m2);

    // Free data arrays
    free(X_train);
    free(X_test);

    return 0;
}



