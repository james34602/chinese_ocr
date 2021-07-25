#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <locale.h>
#include "densenet.h"
#include "cpthread.h"
#include "chineseocr.h"
//#include <vld.h>
#include "misc.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#include "chineseocr.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "base64Text.h"
#include "matrix.h"
#define DEFHEIGHT (32)
typedef struct
{
	int offsetFrom, offsetTo, *nb_filter, *pooledWidth2, *pooledHeight2, *whProd, *whProd2, *width, *height, *nclass;
	int32_t *res[3];
	short *w, *b;
	unsigned short *argmax;
} densenetThreadData;
enum pt_state
{
	SETUP,
	IDLE,
	WORKING,
	GET_OFF_FROM_WORK
};
typedef struct
{
	enum pt_state state;
	pthread_t pth;
	pthread_cond_t work_cond;
	pthread_mutex_t work_mtx;
	pthread_cond_t boss_cond;
	pthread_mutex_t boss_mtx;
	densenetThreadData dat;
	int mode;
} STFTthreadpool;
#define TASK_NB (8)
#define GENCORE (TASK_NB - 1)
void *task_STFT(void *arg)
{
	STFTthreadpool *info = (STFTthreadpool *)arg;
	densenetThreadData *rb = (densenetThreadData*)&info->dat;
	int s, g, i, j;
	// cond_wait mutex must be locked before we can wait
	pthread_mutex_lock(&(info->work_mtx));
	//	printf("<worker %i> start\n", task);
		// ensure boss is waiting
	pthread_mutex_lock(&(info->boss_mtx));
	// signal to boss that setup is complete
	info->state = IDLE;
	// wake-up signal
	pthread_cond_signal(&(info->boss_cond));
	pthread_mutex_unlock(&(info->boss_mtx));
	while (1)
	{
		pthread_cond_wait(&(info->work_cond), &(info->work_mtx));
		if (GET_OFF_FROM_WORK == info->state)
			break; // kill thread
		if (IDLE == info->state)
			continue; // accidental wake-up
		// do blocking task
		if (info->mode == 0) // emConvS
		{
			for (s = rb->offsetFrom; s < rb->offsetTo; s++)
			{
				convolve2DStridedOffset(rb->res[0], &rb->res[1][s * *rb->whProd2], *rb->width, *rb->height, &rb->w[s * 5 * 5], 5, 5, 2, 1);
				for (i = 0; i < *rb->whProd2; i++)
					rb->res[1][s * *rb->whProd2 + i] = sround11(rb->res[1][s * *rb->whProd2 + i]);
			}
		}
		else if (info->mode == 1) // em8_16
		{
			for (s = rb->offsetFrom; s < rb->offsetTo; s++)
			{
				convolve2D3x3(&rb->res[0][0], &rb->res[1][(s + *rb->nb_filter) * *rb->whProd], *rb->width, *rb->height, &rb->w[s * (3 * 3 * *rb->nb_filter)]);
				for (g = 1; g < *rb->nb_filter; g++)
					convolve2D3x3Acc(&rb->res[0][g * *rb->whProd], &rb->res[1][(s + *rb->nb_filter) * *rb->whProd], *rb->width, *rb->height, &rb->w[s * (3 * 3 * *rb->nb_filter) + (3 * 3 * g)]);
				for (i = 0; i < *rb->whProd; i++)
				{
					//printf("%d %1.7f\n", i, res[1][(s + nb_filter) * whProd + i] / 2047.0f);
					rb->res[1][(s + *rb->nb_filter) * *rb->whProd + i] = sround11(rb->res[1][(s + *rb->nb_filter) * *rb->whProd + i]) + rb->b[s];
				}
			}
		}
		else if (info->mode == 2) // emTrans
		{
			for (s = rb->offsetFrom; s < rb->offsetTo; s++)
			{
				for (i = 0; i < *rb->whProd; i++)
					rb->res[2][i] = rb->res[0][i] * rb->w[s * *rb->nb_filter];
				for (g = 1; g < *rb->nb_filter; g++)
					for (i = 0; i < *rb->whProd; i++)
						rb->res[2][i] += rb->res[0][g * *rb->whProd + i] * rb->w[s * *rb->nb_filter + g];
				for (i = 0; i < *rb->whProd; i++)
					rb->res[2][i] = sround11(rb->res[2][i]);
				avgpooling2D(rb->res[2], &rb->res[1][s * *rb->whProd2], *rb->width, *rb->height, 2, 2, 2, 2);
			}
		}
		else if (info->mode == 3)
		{
			for (j = rb->offsetFrom; j < rb->offsetTo; j++)
			{
				rb->argmax[j] = 0;
				int32_t sum = 0;
				for (i = 0; i < *rb->pooledHeight2; i++)
				{
					for (s = 0; s < *rb->nb_filter; s++)
						sum += ((int32_t)rb->w[0 * (*rb->nb_filter * *rb->pooledHeight2) + (*rb->nb_filter * i) + s] * rb->res[0][s * *rb->whProd + i * *rb->pooledWidth2 + j]);
				}
				int32_t maxV = sum + rb->b[0];
				for (g = 1; g < *rb->nclass; g++)
				{
					sum = 0;
					for (i = 0; i < *rb->pooledHeight2; i++)
					{
						for (s = 0; s < *rb->nb_filter; s++)
							sum += ((int32_t)rb->w[g * (*rb->nb_filter * *rb->pooledHeight2) + (*rb->nb_filter * i) + s] * rb->res[0][s * *rb->whProd + i * *rb->pooledWidth2 + j]);
					}
					sum = sum + rb->b[g];
					if (sum > maxV)
					{
						maxV = sum;
						rb->argmax[j] = g;
					}
				}
			}
		}
		// ensure boss is waiting
		pthread_mutex_lock(&(info->boss_mtx));
		// indicate that job is done
		info->state = IDLE;
		// wake-up signal
		pthread_cond_signal(&(info->boss_cond));
		pthread_mutex_unlock(&(info->boss_mtx));
	}
	pthread_mutex_unlock(&(info->work_mtx));
	pthread_exit(NULL);
	return 0;
}
void task_start(STFTthreadpool info[TASK_NB], size_t task, size_t mode)
{
	info[task].mode = mode;
	// ensure worker is waiting
	pthread_mutex_lock(&(info[task].work_mtx));
	// set job information & state
	info[task].state = WORKING;
	// wake-up signal
	pthread_cond_signal(&(info[task].work_cond));
	pthread_mutex_unlock(&(info[task].work_mtx));
}
void task_wait(STFTthreadpool info[TASK_NB], size_t task)
{
	while (1)
	{
		pthread_cond_wait(&(info[task].boss_cond), &(info[task].boss_mtx));
		if (IDLE == info[task].state)
			break;
	}
}
void thread_initSpleeter4s(STFTthreadpool info[TASK_NB])
{
	int i;
	for (i = 0; i < TASK_NB; i++)
	{
		info[i].state = SETUP;
		pthread_cond_init(&(info[i].work_cond), NULL);
		pthread_mutex_init(&(info[i].work_mtx), NULL);
		pthread_cond_init(&(info[i].boss_cond), NULL);
		pthread_mutex_init(&(info[i].boss_mtx), NULL);
		pthread_mutex_lock(&(info[i].boss_mtx));
		pthread_create(&info[i].pth, NULL, task_STFT, (void *)&info[i]);
		task_wait(info, i);
	}
}
void thread_exitSpleeter4s(STFTthreadpool info[TASK_NB])
{
	for (int i = 0; i < TASK_NB; i++)
	{
		// ensure the worker is waiting
		pthread_mutex_lock(&(info[i].work_mtx));
		info[i].state = GET_OFF_FROM_WORK;
		// wake-up signal
		pthread_cond_signal(&(info[i].work_cond));
		pthread_mutex_unlock(&(info[i].work_mtx));
		// wait for thread to exit
		pthread_join(info[i].pth, NULL);
		pthread_mutex_destroy(&(info[i].work_mtx));
		pthread_cond_destroy(&(info[i].work_cond));
		pthread_mutex_unlock(&(info[i].boss_mtx));
		pthread_mutex_destroy(&(info[i].boss_mtx));
		pthread_cond_destroy(&(info[i].boss_cond));
	}
}
int sub_dense_block(int32_t *res[3], int width, int height, short *bn, short *w, short *b, int nb_filter, int growth_rate, int threads, STFTthreadpool threadNN[TASK_NB])
{
	int whProd = width * height;
	int s, g, i;
	for (s = 0; s < nb_filter; s++)
	{
		for (i = 0; i < whProd; i++)
		{
			int32_t normalized = (sround11((int32_t)bn[nb_filter * 1 + s] * res[1][s * whProd + i]) - (int32_t)bn[nb_filter * 2 + s]) + (int32_t)bn[s];
			//printf("%d %1.6f %1.6f\n", i, (sround11((int32_t)bn[nb_filter * 1 + s] * (int32_t)res[1][s * whProd + i]) - (int32_t)bn[nb_filter * 2 + s]) / 2047.0f, normalized / 2047.0f);
			if (normalized < 0)
				res[0][s * whProd + i] = 0;
			else
				res[0][s * whProd + i] = normalized; // ReLU
		}
	}
	int genCores = threads - 1;
	for (i = 0; i < genCores; i++)
	{
		threadNN[i].dat.nb_filter = &nb_filter;
		threadNN[i].dat.whProd = &whProd;
		threadNN[i].dat.width = &width;
		threadNN[i].dat.height = &height;
		threadNN[i].dat.res[0] = res[0];
		threadNN[i].dat.res[1] = res[1];
		threadNN[i].dat.w = w;
		threadNN[i].dat.b = b;
	}
	int taskPerThread1 = growth_rate / threads;
	int leftOver = growth_rate - taskPerThread1 * threads;
	threadNN[0].dat.offsetFrom = taskPerThread1 + leftOver;
	for (i = 0; i < genCores - 1; i++)
	{
		threadNN[i].dat.offsetTo = threadNN[0].dat.offsetFrom + threadNN[i].dat.offsetFrom;
		threadNN[i + 1].dat.offsetFrom = threadNN[i].dat.offsetTo;
	}
	threadNN[genCores - 1].dat.offsetTo = threadNN[genCores - 1].dat.offsetFrom + threadNN[0].dat.offsetFrom;
	for (i = 0; i < genCores; i++)
		task_start(threadNN, i, 1);
	for (s = 0; s < threadNN[0].dat.offsetFrom; s++)
	{
		convolve2D3x3(&res[0][0], &res[1][(s + nb_filter) * whProd], width, height, &w[s * (3 * 3 * nb_filter)]);
		for (g = 1; g < nb_filter; g++)
			convolve2D3x3Acc(&res[0][g * whProd], &res[1][(s + nb_filter) * whProd], width, height, &w[s * (3 * 3 * nb_filter) + (3 * 3 * g)]);
		for (i = 0; i < whProd; i++)
		{
			//printf("%d %1.7f\n", i, res[1][(s + nb_filter) * whProd + i] / 2047.0f);
			res[1][(s + nb_filter) * whProd + i] = sround11(res[1][(s + nb_filter) * whProd + i]) + b[s];
		}
	}
	for (i = 0; i < genCores; i++)
		task_wait(threadNN, i);
	return nb_filter + growth_rate;
}
int transition_block(int32_t *res[3], int width, int height, int outWidth, int outHeight, short *bn, short *w, int dim_convert, int nb_filter, int threads, STFTthreadpool threadNN[TASK_NB])
{
	int whProd = width * height;
	int whProd2 = outWidth * outHeight;
	int s, g, i;
	for (s = 0; s < nb_filter; s++)
	{
		for (i = 0; i < whProd; i++)
		{
			int32_t normalized = (sround11((int32_t)bn[nb_filter * 1 + s] * res[1][s * whProd + i]) - (int32_t)bn[nb_filter * 2 + s]) + (int32_t)bn[s];
			//printf("%d %1.6f %1.6f\n", i, (sround11((int32_t)bn[nb_filter * 1 + s] * (int32_t)res[1][s * whProd + i]) - (int32_t)bn[nb_filter * 2 + s]) / 2047.0f, normalized / 2047.0f);
			if (normalized < 0)
				res[0][s * whProd + i] = 0;
			else
				res[0][s * whProd + i] = normalized; // ReLU
		}
	}
	int genCores = threads - 1;
	for (i = 0; i < genCores; i++)
	{
		threadNN[i].dat.nb_filter = &nb_filter;
		threadNN[i].dat.whProd = &whProd;
		threadNN[i].dat.whProd2 = &whProd2;
		threadNN[i].dat.width = &width;
		threadNN[i].dat.height = &height;
		threadNN[i].dat.res[0] = res[0];
		threadNN[i].dat.res[1] = res[1];
		threadNN[i].dat.res[2] = res[2] + whProd * (i + 1);
		threadNN[i].dat.w = w;
	}
	int taskPerThread1 = dim_convert / threads;
	int leftOver = dim_convert - taskPerThread1 * threads;
	threadNN[0].dat.offsetFrom = taskPerThread1 + leftOver;
	for (i = 0; i < genCores - 1; i++)
	{
		threadNN[i].dat.offsetTo = threadNN[0].dat.offsetFrom + threadNN[i].dat.offsetFrom;
		threadNN[i + 1].dat.offsetFrom = threadNN[i].dat.offsetTo;
	}
	if (genCores)
		threadNN[genCores - 1].dat.offsetTo = threadNN[genCores - 1].dat.offsetFrom + threadNN[0].dat.offsetFrom;
	for (i = 0; i < genCores; i++)
		task_start(threadNN, i, 2);
	for (s = 0; s < threadNN[0].dat.offsetFrom; s++)
	{
		for (i = 0; i < whProd; i++)
			res[2][i] = res[0][i] * w[s * nb_filter];
		for (g = 1; g < nb_filter; g++)
			for (i = 0; i < whProd; i++)
				res[2][i] += res[0][g * whProd + i] * w[s * nb_filter + g];
		for (i = 0; i < whProd; i++)
			res[2][i] = sround11(res[2][i]);
		avgpooling2D(res[2], &res[1][s * whProd2], width, height, 2, 2, 2, 2);
		//for (i = 0; i < whProd2; i++)
		//	printf("%d %1.6f\n", i, res[1][i] / 2047.0f);
	}
	for (i = 0; i < genCores; i++)
		task_wait(threadNN, i);
	return dim_convert;
}
/*	// Weights averaging
	densenet *net = (densenet*)malloc(sizeof(densenet));
	FILE *fp = fopen("simplifiedChineseBest_weights_densenet_03_5_27.dat", "rb");
	fread(net, 1, sizeof(densenet), fp);
	fclose(fp);
	densenet *net2 = (densenet*)malloc(sizeof(densenet));
	fp = fopen("traditionalChineseBestweights_densenet_05_1_59.dat", "rb");
	fread(net2, 1, sizeof(densenet), fp);
	fclose(fp);
	float *ptr1 = (float*)net;
	float *ptr2 = (float*)net2;
	float alpha = 0.259775146842f;
	float minusAlpha = 1.0f - alpha;
	for (i = 0; i < sizeof(densenet) / sizeof(float); i++)
	{
		ptr1[i] = ptr1[i] * alpha + ptr2[i] * minusAlpha;
	}
	fp = fopen("chineseGeneralweights_densenet_04_1_10.dat", "wb");
	fwrite(net, 1, sizeof(densenet), fp);
	fclose(fp);*/
typedef struct
{
	// Constant, must not be changed
	int maxThreads;
	int defImgCap, defStrCap;
	wchar_t wideCh[5990];
	wchar_t *unicodeString;
	int inuse, capacity;
	int maxResizedImgW;
	densenet_quantized net;
	STFTthreadpool threadNN[TASK_NB];
	unsigned char *img_A;
	int32_t *res[3];
} ChineseOCR;
void InitChineseOCR(void *thiz, unsigned char *network, int defaultImgW, int defaultStringCapacity, int threads)
{
	ChineseOCR *ocr = (ChineseOCR*)thiz;
	if (threads < 1)
		ocr->maxThreads = 1;
	else if (threads > 8)
		ocr->maxThreads = 8;
	else
		ocr->maxThreads = threads;
	// Constant setup
	ocr->defImgCap = defaultImgW;
	ocr->defStrCap = defaultStringCapacity;
	// Other NN
	memcpy(&ocr->net, network, sizeof(densenet_quantized));
	ocr->inuse = 0;
	ocr->capacity = defaultStringCapacity;
	ocr->maxResizedImgW = defaultImgW;
	const int poolSize = 2;
	const int poolStride = 2;
	int whProd = defaultImgW * DEFHEIGHT;
	int whProd2 = defaultImgW * (DEFHEIGHT >> 1);
	int featureTransformedWidth = defaultImgW >> 1;
	int pooledWidth1 = ((featureTransformedWidth - poolSize) / poolStride) + 1;
	int pooledWidth2 = ((pooledWidth1 - poolSize) / poolStride) + 1;
	ocr->res[0] = (int32_t*)malloc(whProd2 * 128 * 2 * sizeof(int32_t));
	ocr->res[1] = (int32_t*)malloc(whProd2 * 128 * 2 * sizeof(int32_t));
	ocr->res[2] = (int32_t*)malloc(max(whProd2 * ocr->maxThreads * sizeof(int32_t), (pooledWidth2 * sizeof(unsigned short))));
	ocr->img_A = (unsigned char*)malloc(defaultImgW * 2 * DEFHEIGHT * sizeof(unsigned char));
	// Thread
	thread_initSpleeter4s(ocr->threadNN);
	// String
	base64_decodestate base64;
	base64_init_decodestate(&base64);
	int encodedLen = 23869;
	char *base64Decoded = (char*)malloc(((4 * encodedLen / 3) + 3) & ~3);
	memset(base64Decoded, 0, ((4 * encodedLen / 3) + 3) & ~3);
	int len = base64_decode_block(textCatagoryEncoded, encodedLen, base64Decoded, &base64);
	setlocale(LC_ALL, "zh_CN.utf8");
	mbstowcs(ocr->wideCh, base64Decoded, 5990);
	free(base64Decoded);
	ocr->unicodeString = (wchar_t*)malloc(ocr->capacity * sizeof(wchar_t));
	memset(ocr->unicodeString, 0, ocr->capacity * sizeof(wchar_t));
}
void FreeChineseOCR(void *thiz)
{
	ChineseOCR *ocr = (ChineseOCR*)thiz;
	for (int i = 0; i < TASK_NB; i++)
	{
		if (ocr->threadNN[i].state == WORKING)
			task_wait(ocr->threadNN, i);
	}
	thread_exitSpleeter4s(ocr->threadNN);
	free(ocr->res[0]);
	free(ocr->res[1]);
	free(ocr->res[2]);
	free(ocr->img_A);
	free(ocr->unicodeString);
}
void* ChineseOCRRetUnicodePointer(void *thiz, size_t *in_use)
{
	ChineseOCR *ocr = (ChineseOCR*)thiz;
	*in_use = ocr->inuse * sizeof(wchar_t);
	return (void*)ocr->unicodeString;
}
size_t ChineseOCRRetOCRAllocationSize()
{
	return sizeof(ChineseOCR);
}
void ChineseOCRResetAllContainers(void *thiz)
{
	ChineseOCR *ocr = (ChineseOCR*)thiz;
	// Image container reset
	free(ocr->res[0]);
	free(ocr->res[1]);
	free(ocr->res[2]);
	free(ocr->img_A);
	const int poolSize = 2;
	const int poolStride = 2;
	int whProd = ocr->defImgCap * DEFHEIGHT;
	int whProd2 = ocr->defImgCap * (DEFHEIGHT >> 1);
	int featureTransformedWidth = ocr->defImgCap >> 1;
	int pooledWidth1 = ((featureTransformedWidth - poolSize) / poolStride) + 1;
	int pooledWidth2 = ((pooledWidth1 - poolSize) / poolStride) + 1;
	ocr->res[0] = (int32_t*)malloc(whProd2 * 128 * 2 * sizeof(int32_t));
	ocr->res[1] = (int32_t*)malloc(whProd2 * 128 * 2 * sizeof(int32_t));
	ocr->res[2] = (int32_t*)malloc(max(whProd2 * ocr->maxThreads * sizeof(int32_t), (pooledWidth2 * sizeof(unsigned short))));
	ocr->img_A = (unsigned char*)malloc(ocr->defImgCap * 2 * DEFHEIGHT * sizeof(unsigned char));
	ocr->maxResizedImgW = ocr->defImgCap;
	// String container reset
	ocr->inuse = 0;
	ocr->capacity = ocr->defStrCap;
	free(ocr->unicodeString);
	ocr->unicodeString = (wchar_t*)malloc(ocr->capacity * sizeof(wchar_t));
	memset(ocr->unicodeString, 0, ocr->capacity * sizeof(wchar_t));
}
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
inline int32_t map(int32_t x, int32_t in_min, int32_t in_max)
{
	return (x - in_min) * (int32_t)((((int64_t)1 << (int64_t)32) / (int64_t)(in_max - in_min)) >> 17);
}
void writeNChannelsImg2Folder(int32_t *x, int ch, int w, int h, char *foldername, int32_t linGain)
{
	int nError = 0;
#if defined(_WIN32)
	nError = _mkdir(foldername); // can be used on Windows
#else
	mode_t nMode = 0733; // UNIX style permissions
	nError = mkdir(foldername, nMode); // can be used on non-Windows
#endif
	int s, i;
	int whProd = w * h;
	unsigned char *y = (unsigned char*)malloc(w * h * sizeof(unsigned char));
	int currentMaxCharLen = 4;
	char *filename = (char*)malloc(currentMaxCharLen * sizeof(char));
	for (s = 0; s < ch; s++)
	{
		int32_t mmax = x[s * whProd];
		int32_t mmin = x[s * whProd];
		for (i = 1; i < w * h; i++)
		{
			int32_t val;
			if (x[s * whProd + i] >= 0)
				val = x[s * whProd + i];
			else
				val = -x[s * whProd + i];
			if (val > mmax)
				mmax = val;
			if (val < mmin)
				mmin = val;
		}
		for (i = 0; i < w * h; i++)
		{
			int32_t val;
			if (mmin != mmax)
				val = (map(x[s * whProd + i], mmin, mmax) >> 7) * linGain;
			else
				val = (x[s * whProd + i] >> 7) * linGain;
			if (val > 255)
				val = 255;
			if (val < 0)
				val = 0;
			y[i] = (unsigned char)val;
		}
		size_t bufsz = snprintf(NULL, 0, "%s_%d.png", foldername, s);
		if ((bufsz + 1) > currentMaxCharLen)
		{
			currentMaxCharLen = bufsz + 1;
			filename = (char*)realloc(filename, currentMaxCharLen);
		}
		snprintf(filename, bufsz + 1, "%s/%d.png", foldername, s);
		stbi_write_png(filename, w, h, 1, y, w);
	}
	free(filename);
	free(y);
}
void RecognizeImage(void *thiz, unsigned char *imgUnprepared, int32_t inWidth, int32_t height)
{
	ChineseOCR *ocr = (ChineseOCR*)thiz;
	int s, g, i, j;
	int32_t width = ((int64_t)inWidth << 32) / (((int64_t)height << 32) / (int64_t)DEFHEIGHT);
	if (width % 2)
		width++;
	const int nclass = 5990;
	const int poolSize = 2;
	const int poolStride = 2;
	int whProd = width * DEFHEIGHT;
	int featureTransformedWidth = width >> 1;
	int featureTransformedHeight = DEFHEIGHT >> 1;
	int whProd2 = featureTransformedWidth * featureTransformedHeight;
	int pooledWidth1 = ((featureTransformedWidth - poolSize) / poolStride) + 1;
	int pooledHeight1 = ((featureTransformedHeight - poolSize) / poolStride) + 1;
	int whProd3 = pooledWidth1 * pooledHeight1;
	int pooledWidth2 = ((pooledWidth1 - poolSize) / poolStride) + 1;
	int pooledHeight2 = ((pooledHeight1 - poolSize) / poolStride) + 1;
	int whProd4 = pooledWidth2 * pooledHeight2;

	int nb_filter = 72;
	int growth_rate = 8;
	int dim_convert = 144;
	if (ocr->maxResizedImgW < width)
	{
		free(ocr->res[0]);
		free(ocr->res[1]);
		free(ocr->res[2]);
		ocr->res[0] = (int32_t*)malloc(whProd2 * 128 * 2 * sizeof(int32_t));
		ocr->res[1] = (int32_t*)malloc(whProd2 * 128 * 2 * sizeof(int32_t));
		ocr->res[2] = (int32_t*)malloc(max(whProd2 * ocr->maxThreads * sizeof(int32_t), (pooledWidth2 * sizeof(unsigned short))));
		free(ocr->img_A);
		ocr->img_A = (unsigned char*)malloc(width * DEFHEIGHT * sizeof(unsigned char));
		ocr->maxResizedImgW = width;
	}
	int ret = stbir_resize_uint8(imgUnprepared, inWidth, height, inWidth * sizeof(unsigned char), ocr->img_A, width, DEFHEIGHT, width * sizeof(unsigned char), 1);
	for (i = 0; i < width * DEFHEIGHT; i++)
		ocr->res[0][i] = (((int32_t)ocr->img_A[i]) << 3) - 1023;
	int genCores = ocr->maxThreads - 1;
	int heightPtr = DEFHEIGHT;
	for (i = 0; i < genCores; i++)
	{
		ocr->threadNN[i].dat.width = &width;
		ocr->threadNN[i].dat.height = &heightPtr;
		ocr->threadNN[i].dat.nb_filter = &nb_filter;
		ocr->threadNN[i].dat.whProd2 = &whProd2;
		ocr->threadNN[i].dat.res[0] = ocr->res[0];
		ocr->threadNN[i].dat.res[1] = ocr->res[1];
		ocr->threadNN[i].dat.w = ocr->net.featureTransform_convWeight;
	}
	int taskPerThread1 = nb_filter / ocr->maxThreads;
	int leftOver = nb_filter - taskPerThread1 * ocr->maxThreads;
	ocr->threadNN[0].dat.offsetFrom = taskPerThread1 + leftOver;
	for (i = 0; i < genCores - 1; i++)
	{
		ocr->threadNN[i].dat.offsetTo = ocr->threadNN[i].dat.offsetFrom + taskPerThread1;
		ocr->threadNN[i + 1].dat.offsetFrom = ocr->threadNN[i].dat.offsetTo;
	}
	if (genCores)
		ocr->threadNN[genCores - 1].dat.offsetTo = ocr->threadNN[genCores - 1].dat.offsetFrom + taskPerThread1;
	for (i = 0; i < genCores; i++)
		task_start(ocr->threadNN, i, 0);
	for (s = 0; s < ocr->threadNN[0].dat.offsetFrom; s++)
	{
		convolve2DStridedOffset(ocr->res[0], &ocr->res[1][s * whProd2], width, DEFHEIGHT, &ocr->net.featureTransform_convWeight[s * 5 * 5], 5, 5, 2, 1);
		for (i = 0; i < whProd2; i++)
			ocr->res[1][s * whProd2 + i] = sround11(ocr->res[1][s * whProd2 + i]);
		//printInt32MatrixFile("f2.txt", fsa, featureTransformedWidth, featureTransformedHeight);
	}
	for (i = 0; i < genCores; i++)
		task_wait(ocr->threadNN, i);
    //writeNChannelsImg2Folder(ocr->res[1], nb_filter, featureTransformedWidth, featureTransformedHeight, "features1", 1);
	nb_filter = sub_dense_block(ocr->res, featureTransformedWidth, featureTransformedHeight, ocr->net.dense1_ly1_batchNorm, ocr->net.dense1_ly1_convWeight, ocr->net.dense1_ly1_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, featureTransformedWidth, featureTransformedHeight, ocr->net.dense1_ly2_batchNorm, ocr->net.dense1_ly2_convWeight, ocr->net.dense1_ly2_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, featureTransformedWidth, featureTransformedHeight, ocr->net.dense1_ly3_batchNorm, ocr->net.dense1_ly3_convWeight, ocr->net.dense1_ly3_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, featureTransformedWidth, featureTransformedHeight, ocr->net.dense1_ly4_batchNorm, ocr->net.dense1_ly4_convWeight, ocr->net.dense1_ly4_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, featureTransformedWidth, featureTransformedHeight, ocr->net.dense1_ly5_batchNorm, ocr->net.dense1_ly5_convWeight, ocr->net.dense1_ly5_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, featureTransformedWidth, featureTransformedHeight, ocr->net.dense1_ly6_batchNorm, ocr->net.dense1_ly6_convWeight, ocr->net.dense1_ly6_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, featureTransformedWidth, featureTransformedHeight, ocr->net.dense1_ly7_batchNorm, ocr->net.dense1_ly7_convWeight, ocr->net.dense1_ly7_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, featureTransformedWidth, featureTransformedHeight, ocr->net.dense1_ly8_batchNorm, ocr->net.dense1_ly8_convWeight, ocr->net.dense1_ly8_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = transition_block(ocr->res, featureTransformedWidth, featureTransformedHeight, pooledWidth1, pooledHeight1, ocr->net.transition1_batchNorm, ocr->net.transition1_conv1x1Weight, dim_convert, nb_filter, ocr->maxThreads, ocr->threadNN);
    //writeNChannelsImg2Folder(ocr->res[1], nb_filter, pooledWidth1, pooledHeight1, "features2", 1);
	nb_filter = sub_dense_block(ocr->res, pooledWidth1, pooledHeight1, ocr->net.dense2_ly1_batchNorm, ocr->net.dense2_ly1_convWeight, ocr->net.dense2_ly1_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth1, pooledHeight1, ocr->net.dense2_ly2_batchNorm, ocr->net.dense2_ly2_convWeight, ocr->net.dense2_ly2_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth1, pooledHeight1, ocr->net.dense2_ly3_batchNorm, ocr->net.dense2_ly3_convWeight, ocr->net.dense2_ly3_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth1, pooledHeight1, ocr->net.dense2_ly4_batchNorm, ocr->net.dense2_ly4_convWeight, ocr->net.dense2_ly4_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth1, pooledHeight1, ocr->net.dense2_ly5_batchNorm, ocr->net.dense2_ly5_convWeight, ocr->net.dense2_ly5_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth1, pooledHeight1, ocr->net.dense2_ly6_batchNorm, ocr->net.dense2_ly6_convWeight, ocr->net.dense2_ly6_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth1, pooledHeight1, ocr->net.dense2_ly7_batchNorm, ocr->net.dense2_ly7_convWeight, ocr->net.dense2_ly7_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth1, pooledHeight1, ocr->net.dense2_ly8_batchNorm, ocr->net.dense2_ly8_convWeight, ocr->net.dense2_ly8_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = transition_block(ocr->res, pooledWidth1, pooledHeight1, pooledWidth2, pooledHeight2, ocr->net.transition2_batchNorm, ocr->net.transition2_conv1x1Weight, dim_convert, nb_filter, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth2, pooledHeight2, ocr->net.dense3_ly1_batchNorm, ocr->net.dense3_ly1_convWeight, ocr->net.dense3_ly1_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth2, pooledHeight2, ocr->net.dense3_ly2_batchNorm, ocr->net.dense3_ly2_convWeight, ocr->net.dense3_ly2_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth2, pooledHeight2, ocr->net.dense3_ly3_batchNorm, ocr->net.dense3_ly3_convWeight, ocr->net.dense3_ly3_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth2, pooledHeight2, ocr->net.dense3_ly4_batchNorm, ocr->net.dense3_ly4_convWeight, ocr->net.dense3_ly4_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth2, pooledHeight2, ocr->net.dense3_ly5_batchNorm, ocr->net.dense3_ly5_convWeight, ocr->net.dense3_ly5_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth2, pooledHeight2, ocr->net.dense3_ly6_batchNorm, ocr->net.dense3_ly6_convWeight, ocr->net.dense3_ly6_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth2, pooledHeight2, ocr->net.dense3_ly7_batchNorm, ocr->net.dense3_ly7_convWeight, ocr->net.dense3_ly7_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	nb_filter = sub_dense_block(ocr->res, pooledWidth2, pooledHeight2, ocr->net.dense3_ly8_batchNorm, ocr->net.dense3_ly8_convWeight, ocr->net.dense3_ly8_convBias, nb_filter, 8, ocr->maxThreads, ocr->threadNN);
	for (s = 0; s < nb_filter; s++)
	{
		for (i = 0; i < whProd4; i++)
		{
			int32_t normalized = (sround11((int32_t)ocr->net.lastLayer_batchNorm[nb_filter * 1 + s] * ocr->res[1][s * whProd4 + i]) - (int32_t)ocr->net.lastLayer_batchNorm[nb_filter * 2 + s]) + (int32_t)ocr->net.lastLayer_batchNorm[s];
			//printf("%d %1.6f\n", i, normalized / 2047.0f);
			if (normalized < 0)
				ocr->res[0][s * whProd4 + i] = 0;
			else
				ocr->res[0][s * whProd4 + i] = normalized; // ReLU
		}
	}
	unsigned short *argmax = (unsigned short*)ocr->res[2];
	genCores = ocr->maxThreads - 1;
	for (i = 0; i < genCores; i++)
	{
		ocr->threadNN[i].dat.pooledWidth2 = &pooledWidth2;
		ocr->threadNN[i].dat.pooledHeight2 = &pooledHeight2;
		ocr->threadNN[i].dat.nclass = &nclass;
		ocr->threadNN[i].dat.nb_filter = &nb_filter;
		ocr->threadNN[i].dat.whProd = &whProd4;
		ocr->threadNN[i].dat.res[0] = ocr->res[0];
		ocr->threadNN[i].dat.w = ocr->net.fullyConnectedWeight;
		ocr->threadNN[i].dat.b = ocr->net.fullyConnectedBias;
		ocr->threadNN[i].dat.argmax = argmax;
	}
	taskPerThread1 = pooledWidth2 / ocr->maxThreads;
	leftOver = pooledWidth2 - taskPerThread1 * ocr->maxThreads;
	ocr->threadNN[0].dat.offsetFrom = taskPerThread1 + leftOver;
	for (i = 0; i < genCores - 1; i++)
	{
		ocr->threadNN[i].dat.offsetTo = ocr->threadNN[i].dat.offsetFrom + taskPerThread1;
		ocr->threadNN[i + 1].dat.offsetFrom = ocr->threadNN[i].dat.offsetTo;
	}
	if (genCores)
		ocr->threadNN[genCores - 1].dat.offsetTo = ocr->threadNN[genCores - 1].dat.offsetFrom + taskPerThread1;
	for (i = 0; i < genCores; i++)
		task_start(ocr->threadNN, i, 3);
	for (j = 0; j < ocr->threadNN[0].dat.offsetFrom; j++)
	{
		argmax[j] = 0;
		int32_t sum = 0;
		for (i = 0; i < pooledHeight2; i++)
		{
			for (s = 0; s < nb_filter; s++)
				sum += ((int32_t)ocr->net.fullyConnectedWeight[0 * (nb_filter * pooledHeight2) + (nb_filter * i) + s] * ocr->res[0][s * whProd4 + i * pooledWidth2 + j]);
		}
		int32_t maxV = sum + ocr->net.fullyConnectedBias[0];
		for (g = 1; g < nclass; g++)
		{
			sum = 0;
			for (i = 0; i < pooledHeight2; i++)
			{
				for (s = 0; s < nb_filter; s++)
					sum += ((int32_t)ocr->net.fullyConnectedWeight[g * (nb_filter * pooledHeight2) + (nb_filter * i) + s] * ocr->res[0][s * whProd4 + i * pooledWidth2 + j]);
			}
			sum = sum + ocr->net.fullyConnectedBias[g];
			if (sum > maxV)
			{
				maxV = sum;
				argmax[j] = g;
			}
		}
	}
	for (i = 0; i < genCores; i++)
		task_wait(ocr->threadNN, i);

	memset(ocr->unicodeString, 0, ocr->inuse * sizeof(wchar_t));
	ocr->unicodeString[ocr->inuse] = 0;
	ocr->inuse = 0;
	for (i = 0; i < pooledWidth2; i++)
	{
		int allocate = 0;
		if (!i)
		{
			if (argmax[i] != (nclass - 1))
				allocate = 1;
		}
		else if (i == 1)
		{
			if ((argmax[i] != (nclass - 1)) && ((!(argmax[i] == argmax[i - 1]))))
				allocate = 1;
		}
		else
		{
			if ((argmax[i] != (nclass - 1)) && ((!(argmax[i] == argmax[i - 1])) || (argmax[i] == argmax[i - 2])))
				allocate = 1;
		}
		if (allocate)
		{
			if ((ocr->inuse + 2) > ocr->capacity)
			{
				ocr->capacity += 4;
				ocr->unicodeString = (wchar_t*)realloc(ocr->unicodeString, ocr->capacity * sizeof(wchar_t));
				memset(ocr->unicodeString + ocr->inuse, 0, (ocr->capacity - ocr->inuse) * sizeof(wchar_t));
			}
			ocr->unicodeString[ocr->inuse++] = ocr->wideCh[argmax[i]];
		}
	}
}
