#include <stdint.h>
void InitChineseOCR(void *thiz, unsigned char *network, int defaultImgW, int defaultStringCapacity, int threads);
void FreeChineseOCR(void *thiz);
void* ChineseOCRRetUnicodePointer(void *thiz, size_t *in_use);
size_t ChineseOCRRetOCRAllocationSize();
size_t ChineseOCRRetNNAllocationSize();
void ChineseOCRResetAllContainers(void *thiz);
void RecognizeImage(void *thiz, unsigned char *imgUnprepared, int32_t inWidth, int32_t height);
