
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <ctime>
#include <chrono>

#define DEVICE 0
#define MAX_THREADS 5'400

#define RAND_CONST_1 61746235
#define RAND_CONST_2 18234565
#define RAND_CONST_3 14256765

#define SAMPLE_NUM 60'000
#define BATCH_SIZE 20
#define TOTAL_NEURONS 110

#define SIGMOID(x) ( 1/ (1+exp(-x)) )
#define D_SIGMOID(x) ( (exp(-x)) / (pow((1 + (exp(-x))), 2)) ) 

#pragma warning(push)
#pragma warning(disable : 6386)
#pragma warning(disable : 6385)

class timer {
private:
	std::chrono::system_clock::time_point measure_start;
	std::chrono::system_clock::time_point measure_end;
public:
	timer() {}
	void start()
	{
		measure_start = std::chrono::system_clock::now();
	}
	void check(std::string message)
	{
		measure_end = std::chrono::system_clock::now();
		int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(measure_end - measure_start).count());
		std::cout << message << elapsed << std::endl;
	}
};

__host__ void check_cuda_errors(cudaError_t operation_status, std::string error_report, void* ptr = nullptr)
{
	if (operation_status != cudaSuccess)
	{
		std::cerr << error_report << std::endl;
		if (ptr != nullptr) cudaFree(ptr);
		exit(-1);
	}
}

__host__ double random_number(double mean, double sigma)
{
	std::random_device rd;
	std::mt19937 mersenne_twister(rd());
	std::normal_distribution<double> nd(mean, sigma);
	return nd(mersenne_twister);
}

__host__ void init_weights(double* weights, int* layers, int layer_count)
{
	size_t arr_position = 0;
	for (size_t layer = 0; layer < layer_count; ++layer)
	{
		for (size_t w = 0; w < layers[2 * layer] * layers[2 * layer + 1]; ++w)
		{
			weights[arr_position + w] = random_number(0, 1 / sqrt(layers[2 * layer + 1]));
		}
		arr_position += layers[2 * layer] * layers[2 * layer + 1];
	}
}

__host__ void init_biases(double* vector, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		vector[i] = random_number(0, 1.0);
	}
}

__host__ void check_device_properties()
{ 
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, DEVICE);

	std::cout << " Device #" << DEVICE << ": [" << device_properties.name << "] props: " << std::endl; 
	std::cout << " Multiprocessors: " << device_properties.multiProcessorCount << std::endl;
	std::cout << " Max threads per multiprocessor: " << device_properties.maxThreadsPerMultiProcessor << std::endl;
	std::cout << " Max threads per block: " << device_properties.maxThreadsPerBlock << std::endl;
	std::cout << " Max blocks per multiprocessor: " << device_properties.maxBlocksPerMultiProcessor << std::endl;
	std::cout << " Shared memory per block: " << device_properties.sharedMemPerBlock << std::endl;
	std::cout << " Max grid dimensions: [" << device_properties.maxGridSize[0] << " " << device_properties.maxGridSize[1] << " ";
	std::cout << device_properties.maxGridSize[2] << "]" << std::endl;
	std::cout << " Max block dimensions: [" << device_properties.maxThreadsDim[0] << " " << device_properties.maxThreadsDim[1] << " ";
	std::cout << device_properties.maxThreadsDim[2] << "]" << std::endl;
	std::cout << " Max threads total: " << device_properties.multiProcessorCount * device_properties.maxThreadsPerMultiProcessor << std::endl;
	std::cout << " Const memory total: " << device_properties.totalConstMem << std::endl;
	std::cout << " Global memory total: " << device_properties.totalGlobalMem << std::endl;
	std::cout << " Registers per block available: " << device_properties.regsPerBlock << std::endl;
	std::cout << " Registers per multiprocessor available: " << device_properties.regsPerMultiprocessor << std::endl;
	std::cout << " *    *    *    * " << std::endl;
	if (BATCH_SIZE * TOTAL_NEURONS * 3 * sizeof(float) >= device_properties.sharedMemPerBlock)
	{
		std::cout << " Warning: not enough shared memory per block for such large net. Exiting" << std::endl;
		exit(-1);
	}
}

__host__ void read_data(double* input, double* output, size_t sample_count, std::string digits, std::string labels)
{
	std::ifstream mnist_digits;
	std::ifstream mnist_labels;
	mnist_digits.open(digits, std::ios::binary);
	mnist_labels.open(labels, std::ios::in);
	if (!mnist_digits.is_open() || !mnist_labels.is_open())
	{
		std::cout << "Error: at least one of files cannot be open." << std::endl;
		exit(-1);
	}
	mnist_digits.seekg(16);
	uint8_t byte;
	int label;
	for (size_t sample = 0; sample < sample_count; ++sample)
	{
		// Reading image
		for (size_t pix = 0; pix < 28 * 28; ++pix)
		{
			mnist_digits.read(reinterpret_cast<char*>(&byte), sizeof(char));
			//input[sample * 28 * 28 + pix] = static_cast<double>(byte) / 255;
			if (byte != 0)
			{
				input[sample * 28 * 28 + pix] = 1;
			}
			else
			{
				input[sample * 28 * 28 + pix] = 0;
			}
		}
		// Reading label
		mnist_labels >> label;
		for (size_t digit = 0; digit < 10; ++digit)
		{
			output[sample * 10 + digit] = 0.0;
		}
		output[sample * 10 + label] = 1.0;
	}
	mnist_digits.close();
	mnist_labels.close();
}

__host__ void print_sample(double* input, int sample_number)
{
	for (int y = 0; y < 28; ++y)
	{
		for (int x = 0; x < 28; ++x)
		{
			double byte = input[sample_number * 28 * 28 + y * 28 + x];

			if (byte == 0) std::cout << ". ";
			if (byte > 0) std::cout << "@ ";
		}
		std::cout << std::endl;
	}
}

__host__ void checkpoint(int num)
{
	cudaError_t last_error = cudaGetLastError();
	if (last_error == cudaSuccess)
	{
		std::cout << " Checkpoint [" << num << "]: No errors " << std::endl;
	}
	else
	{
		std::cout << " Checkpoint [" << num << "]: Error <" << last_error << ">: " << cudaGetErrorString(last_error) << std::endl;
	}
}

__host__ void save_weights(double* weights, double* biases, int* layers, int layer_count, std::string network_name)
{
	for (size_t layer = 0; layer < layer_count; ++layer)
	{
		std::ofstream outfile;
		std::stringstream filename;
		filename << network_name << "_" << layer << ".txt";
		outfile.open(filename.str(), std::ios::out);
		if (!outfile.is_open()) std::cout << "Cannot create file." << std::endl, exit(-1);

		outfile << layers[2 * layer] << " " << layers[2 * layer + 1] << std::endl;

		int weights_pos = 0, bias_pos = 0;
		for (int i = 0; i < layer; ++i) weights_pos += layers[2 * i] * layers[2 * i + 1], bias_pos += layers[2 * i];

		for (size_t row = 0; row < layers[2 * layer]; ++row)
		{
			for (size_t col = 0; col < layers[2 * layer + 1]; ++col)
			{
				outfile << weights[weights_pos + row * layers[2 * layer + 1] + col] << " ";
			}
			outfile << biases[bias_pos + row] << std::endl;
		}
		outfile.close();
	}
}

#define SAMPLE_ID(a, b) (((RAND_CONST_1 * a + RAND_CONST_2)*(RAND_CONST_3 + b)) % SAMPLE_NUM)

__global__ void learn_kernel(double* train_inp, double* train_out, double* weights, double* biases, int* layers, int layer_num, double learning_rate)
{
	__shared__ float a_buffer[BATCH_SIZE * TOTAL_NEURONS];
	__shared__ float z_buffer[BATCH_SIZE * TOTAL_NEURONS];
	__shared__ float d_buffer[BATCH_SIZE * TOTAL_NEURONS];
	int l, i, j, layer_shift = 0, weights_shift = 0, layer_shift_ = 0;
	float tmp_val;

	// 1. Feeding input forward through net to gain weighted sums and activations

	// A. First layer getting perception from input
	for (j = 0; j < layers[0]; ++j)
	{
		tmp_val = 0.0;
		for (i = 0; i < layers[1]; ++i) tmp_val += weights[layers[1]*j+i]*train_inp[784* SAMPLE_ID(threadIdx.x, blockIdx.x) +i];
		z_buffer[TOTAL_NEURONS * threadIdx.x + j] = tmp_val + biases[j];
		a_buffer[TOTAL_NEURONS * threadIdx.x + j] = SIGMOID(z_buffer[TOTAL_NEURONS * threadIdx.x + j]);
	}
	// B. Other layers:
	for (l = 1; l < layer_num; ++l)
	{
		for (i = 0; i < l; ++i) layer_shift += layers[2 * i], weights_shift += layers[2 * i] * layers[2 * i + 1];
		//for (i = 0; i < l - 1; ++i) layer_shift_ += layers[2 * i];
		layer_shift_ = layer_shift - layers[2 * (i-1)];
		for (j = 0; j < layers[2 * l]; ++j)
		{
			tmp_val = 0.0;
			for (i = 0; i < layers[2 * l + 1]; ++i) tmp_val += weights[weights_shift+layers[2*l+1]*j+i]*a_buffer[threadIdx.x*TOTAL_NEURONS+layer_shift_+i];
			z_buffer[TOTAL_NEURONS * threadIdx.x + layer_shift + j] = tmp_val + biases[layer_shift + j];
			a_buffer[TOTAL_NEURONS * threadIdx.x + layer_shift + j] = SIGMOID(z_buffer[TOTAL_NEURONS * threadIdx.x + layer_shift + j]);
		}
	}
	layer_shift = 0; layer_shift_ = 0; weights_shift = 0;


	// 2. Calculating Lastlayer delta:
	for (i = 0; i < layer_num - 1; ++i) layer_shift += layers[2 * i];
	for (j = 0; j < 10; ++j)
	{
		d_buffer[TOTAL_NEURONS*threadIdx.x+layer_shift+j] = (a_buffer[TOTAL_NEURONS*threadIdx.x+layer_shift+j]-train_out[10* SAMPLE_ID(threadIdx.x, blockIdx.x) +j])*D_SIGMOID(z_buffer[TOTAL_NEURONS*threadIdx.x+layer_shift+j]);
	}
	layer_shift = 0; layer_shift_ = 0; weights_shift = 0;
	
	// 3. Backpropagating error through layers:
	for (l = layer_num - 2; l >= 0; --l)
	{
		//for (i = 0; i < l; ++i) layer_shift += layers[2 * i];
		for (i = 0; i < l + 1; ++i) layer_shift_ += layers[2 * i], weights_shift += layers[2 * i] * layers[2 * i + 1];
		layer_shift = layer_shift_ - layers[2 * (i - 1)];
		for (j = 0; j < layers[2 * l]; ++j)
		{
			tmp_val = 0.0;
			for (i = 0; i < layers[2*(l+1)]; ++i) tmp_val += d_buffer[TOTAL_NEURONS*threadIdx.x+layer_shift_+i]*weights[weights_shift+layers[2*(l+1)+1]*i+j];
			d_buffer[TOTAL_NEURONS * threadIdx.x + layer_shift + j] = tmp_val * D_SIGMOID(z_buffer[TOTAL_NEURONS*threadIdx.x + layer_shift + j]);
		}
	}
	layer_shift = 0; layer_shift_ = 0; weights_shift = 0;

	// 4. Updating weights and biases
	// A'. Biases
	for (j = 0; j < layers[0]; ++j)
	{
		biases[j] -= (learning_rate / BATCH_SIZE) * d_buffer[TOTAL_NEURONS * threadIdx.x + j];
	}
	// B'. Weights
	for (i = 0; i < layers[0]; ++i)
	{
		for (j = 0; j < layers[1]; ++j)
		{
			weights[i * layers[1] + j] -= (learning_rate / BATCH_SIZE) * d_buffer[TOTAL_NEURONS * threadIdx.x + i] * train_inp[28 * 28 * SAMPLE_ID(threadIdx.x, blockIdx.x) + j];
		}
	}
	for (l = 1; l < layer_num; ++l)
	{
		//for (i = 0; i < l - 1; ++i) layer_shift_ += layers[2 * i];
		for (i = 0; i < l; ++i) layer_shift += layers[2 * i], weights_shift += layers[2 * i] * layers[2 * i + 1];
		layer_shift_ = layer_shift - layers[2 * (i-1)];
		// A. Biases
		for (j = 0; j < layers[2 * l]; ++j)
		{
			biases[layer_shift + j] -= (learning_rate / BATCH_SIZE) * d_buffer[TOTAL_NEURONS * threadIdx.x + layer_shift + j];
		}

		// B. Weights
		for (i = 0; i < layers[2 * l]; ++i)
		{
			for (j = 0; j < layers[2 * l + 1]; ++j)
			{
				weights[weights_shift+i*layers[2*l+1]+j] -= (learning_rate/BATCH_SIZE)*d_buffer[TOTAL_NEURONS*threadIdx.x+layer_shift+i]*a_buffer[TOTAL_NEURONS*threadIdx.x+layer_shift_+j];
			}
		}
	}
}

__host__ int main(int argc, char* argv[])
{
	timer T;
	srand(time(nullptr));
	check_cuda_errors(cudaSetDevice(DEVICE), "Error: cannot set device.");
	check_device_properties();
	
	size_t sample_count = SAMPLE_NUM;
    	size_t digit_shape = 28 * 28;
	double learning_rate = 7.0;
	int batch_size = BATCH_SIZE;

	std::string path = "Data";
	std::string digits = "\\train-images.idx3-ubyte";
	std::string labels = "\\train_labels.txt";
	
	double* train_input = new double[(sample_count*digit_shape)];
	double* train_output = new double[(sample_count * 10)];
	double* layers_weights;
	double* layers_biases;
	int* layers;
	
	size_t layer_count = 2;
	layers = new int[layer_count * 2];
	size_t total_weights = 0;
	size_t total_neurons = 0;
	layers[0] = 100; layers[1] = 784; layers[2] = 10; layers[3] = 100;

	for (size_t layer = 0; layer < layer_count; ++layer)
	{
		total_weights += layers[2 * layer] * layers[2 * layer + 1];
		total_neurons += layers[2 * layer];
	}
	
	// Setting grid parameters:
	dim3 grid_dim(sample_count / batch_size, 1, 1);
	dim3 block_dim(batch_size, 1, 1);
	//std::cout << " Block number: " << sample_count / BATCH_SIZE << std::endl;

	// Initializing weights and biases
	layers_weights = new double[total_weights];
	layers_biases = new double[total_neurons];
	init_weights(layers_weights, layers, layer_count);
	init_biases(layers_biases, total_neurons);
	
	save_weights(layers_weights, layers_biases, layers, layer_count, "Cuda_Learned_Net_Start");

	// Reading train data
	read_data(train_input, train_output, sample_count, path + digits, path + labels);
	std::cout << " Data scanned." << std::endl;
	
	// Allocating GPU device memory
	double* device_train_inp;
	double* device_train_out;
	double* device_weights;
	double* device_biases;
	int* device_layers;

	check_cuda_errors(cudaMalloc(&device_layers, 2*layer_count * sizeof(int)), "MemAlloc failed.");
	check_cuda_errors(cudaMalloc(&device_train_inp, sample_count * digit_shape * sizeof(double)), "MemAlloc failed. ");
	check_cuda_errors(cudaMalloc(&device_train_out, sample_count * 10 * sizeof(double)), "MemAlloc failed. ");
	check_cuda_errors(cudaMalloc(&device_weights, total_weights * sizeof(double)), "MemAlloc failed. ");
	check_cuda_errors(cudaMalloc(&device_biases, total_neurons * sizeof(double)), "MemAlloc failed. ");

	checkpoint(1); /// ~~~ error_point

	// Transfering data to GPU device
	cudaMemcpyKind HTD = cudaMemcpyHostToDevice;
	cudaMemcpyKind DTH = cudaMemcpyDeviceToHost;

	T.start();
	check_cuda_errors(cudaMemcpy(device_layers, layers, 2 * layer_count * sizeof(int), HTD), "MemCopy failed.", device_layers);
	check_cuda_errors(cudaMemcpy(device_train_inp, train_input, sample_count * digit_shape * sizeof(double), HTD), "MemCopy failed. ", device_train_inp);
	check_cuda_errors(cudaMemcpy(device_train_out, train_output, sample_count * 10 * sizeof(double), HTD), "MemCopy failed. ", device_train_out);
	check_cuda_errors(cudaMemcpy(device_weights, layers_weights, total_weights*sizeof(double), HTD), "MemCopy failed. ", device_weights);
	check_cuda_errors(cudaMemcpy(device_biases, layers_biases, total_neurons * sizeof(double), HTD), "MemCopy failed. ", device_biases);
	T.check(" Memcopy-1 taken: ");

	checkpoint(2); /// ~~~ error_point

	// Initializing random states and invoking kernel
	T.start();
	/// ==========================================================================================================================
	// double* train_inp, double* train_out, double* weights, double* biases, int* layers, int layer_num, double learning_rate
	learn_kernel <<<grid_dim, block_dim>>> (device_train_inp, device_train_out, device_weights, device_biases, device_layers, layer_count, learning_rate);
	cudaThreadSynchronize();
	/// ==========================================================================================================================
	T.check(" Kernel taken: ");

	checkpoint(3); /// ~~~ error_point

	// Transfering data from GPU device
	T.start();
	check_cuda_errors(cudaMemcpy(train_input, device_train_inp, sample_count * digit_shape * sizeof(double), DTH), "MemCopy failed. ", device_train_inp);
	check_cuda_errors(cudaMemcpy(train_output, device_train_out, sample_count * 10 * sizeof(double), DTH), "MemCopy failed. ", device_train_out);
	check_cuda_errors(cudaMemcpy(layers_weights, device_weights, total_weights * sizeof(double), DTH), "MemCopy failed. ", device_weights);
	check_cuda_errors(cudaMemcpy(layers_biases, device_biases, total_neurons * sizeof(double), DTH), "MemCopy failed. ", device_biases);
	T.check(" Memcopy-2 taken: ");

	checkpoint(4); /// ~~~ error_point

	// Releasing device memory
	check_cuda_errors(cudaFree(device_layers), "MemFree failed.");
	check_cuda_errors(cudaFree(device_train_inp), "MemFree failed. ");
	check_cuda_errors(cudaFree(device_train_out), "MemFree failed. ");
	check_cuda_errors(cudaFree(device_weights), "MemFree failed. ");
	check_cuda_errors(cudaFree(device_biases), "MemFree failed. ");

	save_weights(layers_weights, layers_biases, layers, layer_count, "Cuda_Learned_Net");
	checkpoint(5); /// ~~~ error_point

	// Releasing host memory
	delete[] layers_weights;
	delete[] layers_biases;
	delete[] train_input;
	delete[] layers;
	delete[] train_output;

	return 0;
}

#pragma warning(pop)

