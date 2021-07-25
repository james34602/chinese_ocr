# Chinese handwritten recognition

A demo of neural network based Chinese character recognition system with fixed point arithmetic.

![Screenshot](/screenshots/demo1-fs8.png?raw=true "Screenshot 1")
![Screenshot](/screenshots/demo2-fs8.png?raw=true "Screenshot 2")

- Neural network architecture based on DenseNet
- Three pretrained models(10MB each), each model train for different character sets, which are simplified Chinese and traditional Chinese
- Trained on 1 billion+ characters with tons of fonts and data argumentation to increase accuracy of handwritten character recognition
- Fixed point(Int32) computation
- Heavy multithreading, no SIMD, should be as fast as NN library(GEMM) with SIMD support

GUI interface based on qt-simple-paint repo