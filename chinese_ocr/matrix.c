#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "matrix.h"
void MatrixAllocate(Matrix *mat, int w, int h)
{
	mat->width = w;
	mat->height = h;
	if (w > 0 && h > 0)
		mat->data = (float*)malloc(w * h * sizeof(float));
	else
		mat->data = 0;
}
float get(Matrix *mat, int x, int y)
{
	return mat->data[y * mat->width + x];
}
inline void set(Matrix *mat, int x, int y, float v)
{
	if (x < 0 || y < 0 || x >= mat->width || y >= mat->height)
		return;
	else
		mat->data[y * mat->width + x] = v;
}
void LoadToMatrix(Matrix *mat, int width, int height, unsigned char *to)
{
	MatrixAllocate(mat, width, height);
	for (int i = 0; i < mat->width * mat->height; i++)
		mat->data[i] = (float)to[i] / 255.0f;
}
float** channel_split(unsigned char* buffer, int *width, int *height, int num_channels)
{
	int i, num_frames = *width * *height;
	float **chan_buffers = (float**)malloc(num_channels * sizeof(float*));
	for (i = 0; i < num_channels; i++)
		chan_buffers[i] = (float*)malloc(*width * *height * sizeof(float));
	int samples = num_frames * num_channels;
	for (i = 0; i < samples; i++)
		chan_buffers[(i % num_channels)][i / num_channels] = buffer[i] / 255.0f;
	float **chan_buffersFinal = (float**)malloc(num_channels * sizeof(float*));
	int inc1 = 0, inc2 = 0;
	if (*width % 2)
		inc1 = 1;
	if (*height % 2)
		inc2 = 1;
	for (i = 0; i < num_channels; i++)
		chan_buffersFinal[i] = (float*)calloc((*width + inc1) * (*height + inc2), sizeof(float));
	for (int c = 0; c < num_channels; c++)
		for (int j = 0; j < *width + inc1; j++)
			for (i = 0; i < *height + inc2; i++)
			{
				if (j == *width && i == *height)
					chan_buffersFinal[c][i * (*width + inc1) + j] = chan_buffers[c][(i - 1) * *width + *width - 1];
				else if (j == *width && i != *height)
					chan_buffersFinal[c][i * (*width + inc1) + j] = chan_buffers[c][i * *width + *width - 1];
				else if (j != *width && i == *height)
					chan_buffersFinal[c][i * (*width + inc1) + j] = chan_buffers[c][(*height - 1) * *width + j];
				else
					chan_buffersFinal[c][i * (*width + inc1) + j] = chan_buffers[c][i * *width + j];
			}
	for (int c = 0; c < num_channels; c++)
		free(chan_buffers[c]);
	free(chan_buffers);
	if (*width % 2)
		*width = *width + 1;
	if (*height % 2)
		*height = *height + 1;
	return chan_buffersFinal;
}
unsigned char* channel_join(float** chan_buffers, int num_frames, int num_channels)
{
	unsigned char *buffer = (unsigned char*)malloc(num_frames * num_channels * sizeof(unsigned char));
	for (int i = 0; i < num_frames * num_channels; i++)
	{
		float temp = chan_buffers[i % num_channels][i / num_channels] * 255.0f;
		if (temp > 255.0f)
			buffer[i] = 255;
		else if (temp < 0.0f)
			buffer[i] = 0;
		else
			buffer[i] = (unsigned char)temp;
	}
	return buffer;
}
#ifndef M_PI
#define M_PI 3.141592653589793
#endif
void fspecial_gaussian(double *kernel, const int N, const double sigma)
{
	double mean = (double)(N - 1) / 2.0;
	double sum = 0.0; // For accumulating the kernel values
	for (int x = 0; x < N; ++x)
		for (int y = 0; y < N; ++y)
		{
			kernel[x * N + y] = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2.0 * M_PI * sigma * sigma);
			sum += kernel[x * N + y];
		}
	for (int x = 0; x < N*N; ++x)
		kernel[x] /= sum;
}
/*
Full:
dstWidth = srcWidth + kernelWidth - 1
dstHeight = srcHeight + kernelHeight - 1
Same:
dstWidth = srcWidth
dstHeight = srcHeight
Valid:
dstWidth = srcWidth - kernelWidth + 1;
dstHeight = srcHeight - kernelHeight + 1;
Performance:
Very low
Numerical accuracy:
Very high
*/
void conv2(Matrix *src, Matrix *dst, Matrix *kernel, int shape)
{
	int src_cols = src->width;
	int src_rows = src->height;
	int kernel_cols = kernel->width;
	int kernel_rows = kernel->height;
	int edge_rows = 0, edge_cols = 0;
	int i, j, kernel_i, kernel_j, src_i, src_j;
	float *p_src = NULL;
	float *p_dst = NULL;
	float *p_kernel = NULL;
	float *p_dst_line_i = NULL;
	float *ptr_src_line_i = NULL;
	float *ptr_kernel_line_i = NULL;
	float sum;
	p_src = src->data;
	p_dst = dst->data;
	p_kernel = kernel->data;
	switch (shape)
	{
	case 0:
		dst->height = src_rows + kernel_rows - 1;
		dst->width = src_cols + kernel_cols - 1;
		edge_rows = kernel_rows - 1;
		edge_cols = kernel_cols - 1;
		break;
	case 1:
		dst->height = src_rows;
		dst->width = src_cols;
		edge_rows = (kernel_rows - 1) / 2;
		edge_cols = (kernel_cols - 1) / 2;
		break;
	case 2:
		dst->height = src_rows - kernel_rows + 1;
		dst->width = src_cols - kernel_cols + 1;
		edge_rows = edge_cols = 0;
		break;
	}
	for (i = 0; i < dst->height; i++)
	{
		p_dst_line_i = (float*)(p_dst + dst->width * i);
		for (j = 0; j < dst->width; j++)
		{
			sum = 0.0f;
			kernel_i = kernel_rows - 1 - max(0, edge_rows - i);
			src_i = max(0, i - edge_rows);
			for (; (kernel_i >= 0) && (src_i < src_rows); kernel_i--, src_i++)
			{
				kernel_j = kernel_cols - 1 - max(0, edge_cols - j);
				src_j = max(0, j - edge_cols);
				ptr_src_line_i = (float*)(p_src + src_cols * src_i);
				ptr_kernel_line_i = (float*)(p_kernel + kernel_cols * kernel_i);
				ptr_src_line_i += src_j;
				ptr_kernel_line_i += kernel_j;
				for (; kernel_j >= 0 && src_j < src_cols; kernel_j--, src_j++)
					sum += *ptr_src_line_i++ * *ptr_kernel_line_i--;
			}
			p_dst_line_i[j] = sum;
		}
	}
}
/* Matlab conv2 with 'same'
Performance:
Middle - Low
Numerical accuracy:
Very high
*/
void convolve2D(int32_t* in, int32_t* out, int width, int height, short* kernel, int kernelSizeX, int kernelSizeY)
{
	int i, j, m, n;
	int32_t *inPtr, *inPtr2, *outPtr;
	short *kPtr;
	int32_t acc;
	int kCenterX, kCenterY;
	int rowMin, rowMax; // to check boundary of input array
	int colMin, colMax;
	// find center position of kernel (half of kernel size)
	kCenterX = kernelSizeX >> 1;
	kCenterY = kernelSizeY >> 1;
	// init working  pointers
	inPtr = inPtr2 = &in[width * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
	outPtr = out;
	kPtr = kernel;
	// start convolution
	for (i = 0; i < height; ++i) // number of rows
	{
		// compute the range of convolution, the current row of kernel should be between these
		rowMax = i + kCenterY;
		rowMin = i - height + kCenterY;
		for (j = 0; j < width; ++j)              // number of columns
		{
			// compute the range of convolution, the current column of kernel should be between these
			colMax = j + kCenterX;
			colMin = j - width + kCenterX;
			acc = 0; // set to 0 before accumulate (Clear dst value)
			// flip the kernel and traverse all the kernel values
			// multiply each kernel value with underlying input data
			for (m = 0; m < kernelSizeY; ++m)        // kernel rows
			{
				// check if the index is out of bound of input array
				if (m <= rowMax && m > rowMin)
				{
					for (n = 0; n < kernelSizeX; ++n)
					{
						// check the boundary of array
						if (n <= colMax && n > colMin)
							acc += *(inPtr - n) * *kPtr;
						++kPtr;                     // next kernel
					}
				}
				else
					kPtr += kernelSizeX;            // out of bound, move to next row of kernel
				inPtr -= width;                 // move input data 1 raw up
			}
			*outPtr = acc;
			kPtr = kernel;                          // reset kernel to (0,0)
			inPtr = ++inPtr2;                       // next input
			++outPtr;                               // next output
		}
	}
}
void convolve2DStridedOffset(int32_t* in, int32_t* out, int width, int height, short* kernel, int kernelSizeX, int kernelSizeY, int stride, int offset)
{
	int i, j, m, n;
	int32_t *inPtr, *inPtr2, *outPtr;
	short *kPtr;
	int32_t acc;
	int kCenterX, kCenterY;
	int rowMin, rowMax; // to check boundary of input array
	int colMin, colMax;
	// find center position of kernel (half of kernel size)
	kCenterX = kernelSizeX >> 1;
	kCenterY = kernelSizeY >> 1;
	// init working  pointers
	inPtr = inPtr2 = &in[width * kCenterY + kCenterX + width + offset];  // note that  it is shifted (kCenterX, kCenterY),
	outPtr = out;
	kPtr = kernel;
	// start convolution
	for (i = offset; i < height; i += stride) // number of rows
	{
		// compute the range of convolution, the current row of kernel should be between these
		rowMax = i + kCenterY;
		rowMin = i - height + kCenterY;
		for (j = offset; j < width; j += stride)              // number of columns
		{
			// compute the range of convolution, the current column of kernel should be between these
			colMax = j + kCenterX;
			colMin = j - width + kCenterX;
			acc = 0; // set to 0 before accumulate (Clear dst value)
			// flip the kernel and traverse all the kernel values
			// multiply each kernel value with underlying input data
			for (m = 0; m < kernelSizeY; ++m)        // kernel rows
			{
				// check if the index is out of bound of input array
				if (m <= rowMax && m > rowMin)
				{
					for (n = 0; n < kernelSizeX; ++n)
					{
						// check the boundary of array
						if (n <= colMax && n > colMin)
							acc += *(inPtr - n) * *kPtr;
						++kPtr;                     // next kernel
					}
				}
				else
					kPtr += kernelSizeX;            // out of bound, move to next row of kernel
				inPtr -= width;                 // move input data 1 raw up
			}
			*outPtr = acc;
			kPtr = kernel;                          // reset kernel to (0,0)
			inPtr2 += stride;
			inPtr = inPtr2;                       // next input
			++outPtr;                               // next output
		}
		inPtr2 += (width * (stride - 1));
		inPtr = inPtr2;                       // next input
	}
}
void convolve2D3x3(int32_t* in, int32_t* out, const int width, int const height, short* kernel)
{
	int i, j, m, n;
	int32_t *inPtr, *inPtr2, *outPtr;
	short *kPtr;
	int32_t acc;
	int rowMin, rowMax;
	int colMin, colMax;
	inPtr = inPtr2 = &in[width + 1];
	outPtr = out;
	kPtr = kernel;
	for (i = 0; i < height; ++i)
	{
		rowMax = i + 1;
		rowMin = i - height + 1;
		for (j = 0; j < width; ++j)
		{
			colMax = j + 1;
			colMin = j - width + 1;
			acc = 0;
			for (m = 0; m < 3; ++m)
			{
				if (m <= rowMax && m > rowMin)
				{
					for (n = 0; n < 3; ++n)
					{
						if (n <= colMax && n > colMin)
							acc += *(inPtr - n) * *kPtr;
						++kPtr;
					}
				}
				else
					kPtr += 3;
				inPtr -= width;
			}
			*outPtr = acc;
			kPtr = kernel;
			inPtr = ++inPtr2;
			++outPtr;
		}
	}
}
void convolve2D3x3Acc(int32_t* in, int32_t* out, const int width, int const height, short* kernel)
{
	int i, j, m, n;
	int32_t *inPtr, *inPtr2, *outPtr;
	short *kPtr;
	int32_t acc;
	int rowMin, rowMax;
	int colMin, colMax;
	inPtr = inPtr2 = &in[width + 1];
	outPtr = out;
	kPtr = kernel;
	for (i = 0; i < height; ++i)
	{
		rowMax = i + 1;
		rowMin = i - height + 1;
		for (j = 0; j < width; ++j)
		{
			colMax = j + 1;
			colMin = j - width + 1;
			acc = 0;
			for (m = 0; m < 3; ++m)
			{
				if (m <= rowMax && m > rowMin)
				{
					for (n = 0; n < 3; ++n)
					{
						if (n <= colMax && n > colMin)
							acc += *(inPtr - n) * *kPtr;
						++kPtr;
					}
				}
				else
					kPtr += 3;
				inPtr -= width;
			}
			*outPtr += acc;
			kPtr = kernel;
			inPtr = ++inPtr2;
			++outPtr;
		}
	}
}
/* Optimized 3x3 convolution
Performance:
Very high
Numerical accuracy:
Low
*/
#include <emmintrin.h>
void conv2DSSEOMP3x3(float* in, float* out, const int width, const int height, float* kernel)
{
	int begin = 1 + (width - 1) / 4 * 4;
	if (width % 4 != 0)
		begin -= 4;
	const int start = begin;
	int range = (start - 1) / 16 * 16 + 1;
	int y, x, j, i;
	for (y = 0; y < height; y++)
	{
		// left section from 0 column to 1    	
		for (x = 0; x < 1; x++)
			for (j = -1; j <= 1; j++)
				for (i = -1; i <= 1; i++)
					if (x + i > -1 && x + i<width && y + j>-1 && y + j < height)
						out[x + y * width] += kernel[(1 - i) + (1 - j)*3] * in[(x + i) + (y + j)*width];
		for (x = 1; x < range; x += 16)
		{
			__m128 out_vec = _mm_loadu_ps(out + x + y * width);
			__m128 out_vec1 = _mm_loadu_ps(out + x + 4 + y * width);
			__m128 out_vec2 = _mm_loadu_ps(out + x + 8 + y * width);
			__m128 out_vec3 = _mm_loadu_ps(out + x + 12 + y * width);
			for (j = -1; j <= 1; j++)
			{
				if (y + j < 0 || y + j >= height)
					continue;
				for (i = -1; i <= 1; i++)
				{
					__m128 ker_vec = _mm_load1_ps(kernel + (1 - i) + (1 - j)*3);
					__m128 in_vec = _mm_loadu_ps(in + x + i + (y + j)*width);
					__m128 in_vec1 = _mm_loadu_ps(in + x + 4 + i + (y + j)*width);
					__m128 in_vec2 = _mm_loadu_ps(in + x + 8 + i + (y + j)*width);
					__m128 in_vec3 = _mm_loadu_ps(in + x + 12 + i + (y + j)*width);
					out_vec = _mm_add_ps(out_vec, _mm_mul_ps(ker_vec, in_vec));
					out_vec1 = _mm_add_ps(out_vec1, _mm_mul_ps(ker_vec, in_vec1));
					out_vec2 = _mm_add_ps(out_vec2, _mm_mul_ps(ker_vec, in_vec2));
					out_vec3 = _mm_add_ps(out_vec3, _mm_mul_ps(ker_vec, in_vec3));
				}
			}
			_mm_storeu_ps(out + x + y * width, out_vec);
			_mm_storeu_ps(out + x + 4 + y * width, out_vec1);
			_mm_storeu_ps(out + x + 8 + y * width, out_vec2);
			_mm_storeu_ps(out + x + 12 + y * width, out_vec3);
		}
		for (x = range; x < start; x += 4)
		{
			__m128 out_vec = _mm_loadu_ps(out + x + y * width);
			for (j = -1; j <= 1; j++)
			{
				if (y + j < 0 || y + j >= height)
					continue;
				for (i = -1; i <= 1; i++)
				{
					__m128 in_vec = _mm_loadu_ps(in + x + i + (y + j)*width);
					__m128 ker_vec = _mm_load1_ps(kernel + (1 - i) + (1 - j)*3);
					out_vec = _mm_add_ps(out_vec, _mm_mul_ps(ker_vec, in_vec));
				}
			}
			_mm_storeu_ps(out + x + y * width, out_vec);
		}
		// right section from the starting to the end of the matrix
		for (x = start; x < width; x++)
			for (j = -1; j <= 1; j++)
				for (i = -1; i <= 1; i++)
					if (x + i > -1 && x + i<width && y + j>-1 && y + j < height)
						out[x + y * width] += kernel[(1 - i) + (1 - j) * 3] * in[(x + i) + (y + j)*width];
	}
}
void conv2DSSEFast(float* in, float* out, const int width, const int height, float* kernel, const int kernel_x, const int kernel_y)
{
	int stride = 2;
	// the x coordinate of the kernel's center
	const int kern_cent_X = (kernel_x - 1) / 2;
	// the y coordinate of the kernel's center
	const int kern_cent_Y = (kernel_y - 1) / 2;
	const int kern_size = kern_cent_Y * kern_cent_X;
	int offset = 0;
	if (kernel_x > 5 && kernel_x < 15)
		offset = 1;
	else if (kernel_x >= 15)
		offset = 2;
	int begin = kern_cent_X + (width - kern_cent_X) / 4 * 4 - (4 * offset);
	if (width % 4 != 0)
		begin -= 4;
	const int start = begin;
	int y;
	for (y = 0; y < height; y++) {
		// left section from 0 column to kern_cent_X    	
		for (int x = 0; x < kern_cent_X; x++) {
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
					if (x + i > -1 && x + i<width && y + j>-1 && y + j < height)
						out[x + y * width] += kernel[(kern_cent_X - i) + (kern_cent_Y - j)*kernel_x] * in[(x + i) + (y + j)*width];
				}
			}
		}
		for (int x = kern_cent_X; x < (start - kern_cent_X) / 16 * 16 + kern_cent_X; x += 16) {
			__m128 out_vec = _mm_loadu_ps(out + x + y * width);
			__m128 out_vec1 = _mm_loadu_ps(out + x + 4 + y * width);
			__m128 out_vec2 = _mm_loadu_ps(out + x + 8 + y * width);
			__m128 out_vec3 = _mm_loadu_ps(out + x + 12 + y * width);
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				if (y + j < 0 || y + j >= height)
					continue;
				for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
					__m128 ker_vec = _mm_load1_ps(kernel + (kern_cent_X - i) + (kern_cent_Y - j)*kernel_x);

					__m128 in_vec = _mm_loadu_ps(in + x + i + (y + j)*width);
					__m128 in_vec1 = _mm_loadu_ps(in + x + 4 + i + (y + j)*width);
					__m128 in_vec2 = _mm_loadu_ps(in + x + 8 + i + (y + j)*width);
					__m128 in_vec3 = _mm_loadu_ps(in + x + 12 + i + (y + j)*width);
					out_vec = _mm_add_ps(out_vec, _mm_mul_ps(ker_vec, in_vec));
					out_vec1 = _mm_add_ps(out_vec1, _mm_mul_ps(ker_vec, in_vec1));
					out_vec2 = _mm_add_ps(out_vec2, _mm_mul_ps(ker_vec, in_vec2));
					out_vec3 = _mm_add_ps(out_vec3, _mm_mul_ps(ker_vec, in_vec3));
				}
			}
			_mm_storeu_ps(out + x + y * width, out_vec);
			_mm_storeu_ps(out + x + 4 + y * width, out_vec1);
			_mm_storeu_ps(out + x + 8 + y * width, out_vec2);
			_mm_storeu_ps(out + x + 12 + y * width, out_vec3);
		}
		for (int x = (start - kern_cent_X) / 16 * 16 + kern_cent_X; x < start; x += 4) {
			__m128 out_vec = _mm_loadu_ps(out + x + y * width);
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				if (y + j < 0 || y + j >= height) {
					continue;
				}
				for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
					__m128 in_vec = _mm_loadu_ps(in + x + i + (y + j)*width);
					__m128 ker_vec = _mm_load1_ps(kernel + (kern_cent_X - i) + (kern_cent_Y - j)*kernel_x);
					out_vec = _mm_add_ps(out_vec, _mm_mul_ps(ker_vec, in_vec));
				}
			}
			_mm_storeu_ps(out + x + y * width, out_vec);
		}
		// right section from the starting to the end of the matrix
		for (int x = start; x < width; x++) {
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
					if (x + i > -1 && x + i<width && y + j>-1 && y + j < height)
						out[x + y * width] += kernel[(kern_cent_X - i) + (kern_cent_Y - j)*kernel_x] * in[(x + i) + (y + j)*width];
				}
			}
		}
	}
}
#include <immintrin.h>
void conv2DAVXOMP3x3(float* in, float* out, const int width, const int height, float* kernel)
{
	int begin = 1 + (width - 1) / 8 * 8;
	if (width % 8 != 0)
		begin -= 8;
	const int start = begin;
	int range = (start - 1) / 32 * 32 + 1;
	int y, x, j, i;
#pragma omp parallel for firstprivate(in, out, kernel)
	for (y = 0; y < height; y++)
	{
		// left section from 0 column to 1    	
		for (x = 0; x < 1; x++)
			for (j = -1; j <= 1; j++)
				for (i = -1; i <= 1; i++)
					if (x + i > -1 && x + i<width && y + j>-1 && y + j < height)
						out[x + y * width] += kernel[(1 - i) + (1 - j) * 3] * in[(x + i) + (y + j)*width];
		for (x = 1; x < range; x += 32)
		{
			__m256 out_vec = _mm256_loadu_ps(out + x + y * width);
			__m256 out_vec1 = _mm256_loadu_ps(out + x + 8 + y * width);
			__m256 out_vec2 = _mm256_loadu_ps(out + x + 16 + y * width);
			__m256 out_vec3 = _mm256_loadu_ps(out + x + 24 + y * width);
			for (j = -1; j <= 1; j++)
			{
				if (y + j < 0 || y + j >= height)
					continue;
				for (i = -1; i <= 1; i++)
				{
					__m256 ker_vec = _mm256_broadcast_ss(kernel + (1 - i) + (1 - j) * 3);
					__m256 in_vec = _mm256_loadu_ps(in + x + i + (y + j)*width);
					__m256 in_vec1 = _mm256_loadu_ps(in + x + 8 + i + (y + j)*width);
					__m256 in_vec2 = _mm256_loadu_ps(in + x + 16 + i + (y + j)*width);
					__m256 in_vec3 = _mm256_loadu_ps(in + x + 24 + i + (y + j)*width);
					out_vec = _mm256_add_ps(out_vec, _mm256_mul_ps(ker_vec, in_vec));
					out_vec1 = _mm256_add_ps(out_vec1, _mm256_mul_ps(ker_vec, in_vec1));
					out_vec2 = _mm256_add_ps(out_vec2, _mm256_mul_ps(ker_vec, in_vec2));
					out_vec3 = _mm256_add_ps(out_vec3, _mm256_mul_ps(ker_vec, in_vec3));
				}
			}
			_mm256_storeu_ps(out + x + y * width, out_vec);
			_mm256_storeu_ps(out + x + 8 + y * width, out_vec1);
			_mm256_storeu_ps(out + x + 16 + y * width, out_vec2);
			_mm256_storeu_ps(out + x + 24 + y * width, out_vec3);
		}
		for (x = range; x < start; x += 8)
		{
			__m256 out_vec = _mm256_loadu_ps(out + x + y * width);
			for (j = -1; j <= 1; j++)
			{
				if (y + j < 0 || y + j >= height)
					continue;
				for (i = -1; i <= 1; i++)
				{
					__m256 in_vec = _mm256_loadu_ps(in + x + i + (y + j)*width);
					__m256 ker_vec = _mm256_broadcast_ss(kernel + (1 - i) + (1 - j) * 3);
					out_vec = _mm256_add_ps(out_vec, _mm256_mul_ps(ker_vec, in_vec));
				}
			}
			_mm256_storeu_ps(out + x + y * width, out_vec);
		}
		// right section from the starting to the end of the matrix
		for (x = start; x < width; x++)
			for (j = -1; j <= 1; j++)
				for (i = -1; i <= 1; i++)
					if (x + i > -1 && x + i<width && y + j>-1 && y + j < height)
						out[x + y * width] += kernel[(1 - i) + (1 - j) * 3] * in[(x + i) + (y + j)*width];
	}
}
void conv2DAVXFast(float* in, float* out, const int width, const int height, float* kernel, const int kernel_x, const int kernel_y)
{
	// the x coordinate of the kernel's center
	const int kern_cent_X = (kernel_x - 1) / 2;
	// the y coordinate of the kernel's center
	const int kern_cent_Y = (kernel_y - 1) / 2;
	const int kern_size = kern_cent_Y * kern_cent_X;
	int offset = 0;
	if (kernel_x > 9 && kernel_x < 35)
		offset = 1;
	else if (kernel_x >= 35)
		offset = 2;
	int begin = kern_cent_X + (width - kern_cent_X) / 8 * 8 - (8 * offset);
	if (width % 8 != 0)
		begin -= 8;
	const int start = begin;
	int y;
#pragma omp parallel for firstprivate(in, out, kernel)
	for (y = 0; y < height; y++) {
		// left section from 0 column to kern_cent_X    	
		for (int x = 0; x < kern_cent_X; x++) {
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
					if (x + i > -1 && x + i<width && y + j>-1 && y + j < height)
						out[x + y * width] += kernel[(kern_cent_X - i) + (kern_cent_Y - j)*kernel_x] * in[(x + i) + (y + j)*width];
				}
			}
		}
		for (int x = kern_cent_X; x < (start - kern_cent_X) / 32 * 32 + kern_cent_X; x += 32) {
			__m256 out_vec = _mm256_loadu_ps(out + x + y * width);
			__m256 out_vec1 = _mm256_loadu_ps(out + x + 8 + y * width);
			__m256 out_vec2 = _mm256_loadu_ps(out + x + 16 + y * width);
			__m256 out_vec3 = _mm256_loadu_ps(out + x + 24 + y * width);
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				if (y + j < 0 || y + j >= height)
					continue;
				for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
					__m256 ker_vec = _mm256_broadcast_ss(kernel + (kern_cent_X - i) + (kern_cent_Y - j)*kernel_x);

					__m256 in_vec = _mm256_loadu_ps(in + x + i + (y + j)*width);
					__m256 in_vec1 = _mm256_loadu_ps(in + x + 8 + i + (y + j)*width);
					__m256 in_vec2 = _mm256_loadu_ps(in + x + 16 + i + (y + j)*width);
					__m256 in_vec3 = _mm256_loadu_ps(in + x + 24 + i + (y + j)*width);
					out_vec = _mm256_add_ps(out_vec, _mm256_mul_ps(ker_vec, in_vec));
					out_vec1 = _mm256_add_ps(out_vec1, _mm256_mul_ps(ker_vec, in_vec1));
					out_vec2 = _mm256_add_ps(out_vec2, _mm256_mul_ps(ker_vec, in_vec2));
					out_vec3 = _mm256_add_ps(out_vec3, _mm256_mul_ps(ker_vec, in_vec3));
				}
			}
			_mm256_storeu_ps(out + x + y * width, out_vec);
			_mm256_storeu_ps(out + x + 8 + y * width, out_vec1);
			_mm256_storeu_ps(out + x + 16 + y * width, out_vec2);
			_mm256_storeu_ps(out + x + 24 + y * width, out_vec3);
		}
		for (int x = (start - kern_cent_X) / 32 * 32 + kern_cent_X; x < start; x += 8) {
			__m256 out_vec = _mm256_loadu_ps(out + x + y * width);
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				if (y + j < 0 || y + j >= height) {
					continue;
				}
				for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
					__m256 in_vec = _mm256_loadu_ps(in + x + i + (y + j)*width);
					__m256 ker_vec = _mm256_broadcast_ss(kernel + (kern_cent_X - i) + (kern_cent_Y - j)*kernel_x);
					out_vec = _mm256_add_ps(out_vec, _mm256_mul_ps(ker_vec, in_vec));
				}
			}
			_mm256_storeu_ps(out + x + y * width, out_vec);
		}
		// right section from the starting to the end of the matrix
		for (int x = start; x < width; x++) {
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
					if (x + i > -1 && x + i<width && y + j>-1 && y + j < height)
						out[x + y * width] += kernel[(kern_cent_X - i) + (kern_cent_Y - j)*kernel_x] * in[(x + i) + (y + j)*width];
				}
			}
		}
	}
}
void maxpooling2D(float *map, float *y, int w, int h, int k_w, int k_h, int s_w, int s_h)
{
	int out_row = (w - k_h) / s_h + 1;
	int out_col = (h - k_w) / s_w + 1;
	for (int i = 0; i < out_col; i++)
		for (int j = 0; j < out_row; j++)
		{
			int start_x = j * s_w;
			int start_y = i * s_h;
			float maxVal = -FLT_MAX;
			for (int ii = 0; ii < k_w; ii++)
				for (int jj = 0; jj < k_h; jj++)
				{
					if (map[(start_y + jj) * w + (start_x + ii)] > maxVal)
						maxVal = map[(start_y + jj) * w + (start_x + ii)];
				}
			y[i * out_row + j] = maxVal;
		}
}
void maxpooling2DLayer(float *x, float *y, int c, int w, int h, int poolSize, int stride, int outW, int outH)
{
	for (int s = 0; s < c; s++)
		maxpooling2D(&x[s * w * h], &y[s * outW * outH], w, h, poolSize, poolSize, stride, stride);
}
void avgpooling2D(int32_t *map, int32_t *y, int w, int h, int k_w, int k_h, int s_w, int s_h)
{
	int out_row = (w - k_h) / s_h + 1;
	int out_col = (h - k_w) / s_w + 1;
	int32_t scale = sdiv(1, k_w * k_h);
	for (int i = 0; i < out_col; i++)
		for (int j = 0; j < out_row; j++)
		{
			int start_x = j * s_w;
			int start_y = i * s_h;
			int32_t maxVal = 0;
			for (int ii = 0; ii < k_w; ii++)
				for (int jj = 0; jj < k_h; jj++)
				{
					maxVal += map[(start_y + jj) * w + (start_x + ii)];
				}
			//printf("%1.6f\n", sround11(maxVal * scale) / 2047.0f);
			y[i * out_row + j] = sround11(maxVal * scale);
		}
}