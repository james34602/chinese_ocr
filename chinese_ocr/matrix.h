#ifndef _SURF_MATRIX_H_
#define _SURF_MATRIX_H_
#include <stdint.h>
#define FRACBITS 12
#define FRACBITSM1 (FRACBITS - 1)
#define sround11( x )  (int32_t)( ( (x) + (1<<(FRACBITSM1-1)) ) >> FRACBITSM1 )
#define sdiv(x,b) (((int32_t)(x) << FRACBITSM1) / (b))
typedef struct
{
	int width;
	int height;
	float *data;
} Matrix;
void MatrixAllocate(Matrix *mat, int w, int h);
float get(Matrix *mat, int x, int y);
float** channel_split(unsigned char* buffer, int *width, int *height, int num_channels);
unsigned char* channel_join(float** chan_buffers, int num_frames, int num_channels);
void fspecial_gaussian(double *kernel, const int N, const double sigma);
void conv2(Matrix *src, Matrix *dst, Matrix *kernel, int shape);
void convolve2D(int32_t* in, int32_t* out, int width, int height, short* kernel, int kernelSizeX, int kernelSizeY);
void convolve2DStridedOffset(int32_t* in, int32_t* out, int width, int height, short* kernel, int kernelSizeX, int kernelSizeY, int stride, int offset);
void convolve2D3x3(int32_t* in, int32_t* out, const int width, const int height, short* kernel);
void convolve2D3x3Acc(int32_t* in, int32_t* out, const int width, int const height, short* kernel);
void conv2DSSEOMP3x3(float* in, float* out, const int width, const int height, float* kernel);
void conv2DAVXOMP3x3(float* in, float* out, const int width, const int height, float* kernel);
void conv2DSSEFast(float* in, float* out, const int width, const int height, float* kernel, const int kernel_x, const int kernel_y);
void conv2DAVXFast(float* in, float* out, const int width, const int height, float* kernel, const int kernel_x, const int kernel_y);
void maxpooling2DLayer(float *x, float *y, int c, int w, int h, int poolSize, int stride, int outW, int outH);
void avgpooling2D(int32_t *map, int32_t *y, int w, int h, int k_w, int k_h, int s_w, int s_h);
#endif