#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include "helper_functions.h"    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check


// Define the files that are to be save and the reference images for validation
const char *imageFilename = "NewsPaper1080.pgm";

// Texture reference for 2D filter and image
//texture<float, 2, cudaReadModeElementType> tex;
//texture<float, 2, cudaReadModeElementType> texi;

const unsigned int MAX_FILTER_SIZE = 10;//79;
__device__ __constant__ float d_cFilterKernel[ MAX_FILTER_SIZE * MAX_FILTER_SIZE ];
__device__ __constant__ float d_cFilterKernel1[ MAX_FILTER_SIZE * MAX_FILTER_SIZE ];
__device__ __constant__ float d_cFilterKernel2[ MAX_FILTER_SIZE * MAX_FILTER_SIZE ];

//function declarations
//void createfilt(char a, float** arr, int size);
void createfilt2(char a, float* arr, int size);
//void convolve(float * output, float* input, float* filt, int height, int width, int k_size, char type);
void GetMag(float * output, float* imx, float* imy, int height, int width);
//void GetDir(float * output, float* imx, float* imy, int height, int width);
void NMSuppress(float* output, float* G,float* after_Gx, float* after_Gy, float thr_min, float thr_max, int width, int height);
void Hysteresis(float* output, float* nms,float* after_Gx, float* after_Gy, float thr_min, float thr_max, int width, int height);

//kernel function
__global__ void conv_kern(float* output, const float* input, int height, int width, int val, char a){

int m = blockDim.x*blockIdx.x + threadIdx.x;
int n = blockDim.y*blockIdx.y + threadIdx.y;
float* filt;

if(a == 'g')
filt = d_cFilterKernel;
else if (a == 'x')
filt = d_cFilterKernel1;
else if (a == 'y')
filt = d_cFilterKernel2;

   float accumulation = 0;
   for (int i = m-(val/2); i < (m+val/2+1); i++){//k
     for (int j = n-(val/2); j < (n+val/2+1); j++ ) {//m
       //if((m+i >= 0 && m+i < height) && (n+j >=0 && n+j< width) ){
       if(i>=0 && i<width && j>=0 && j< height){
       float k = input[(i)*height + (j)];//input(m+i, n+j); right by a b
       accumulation = accumulation + (k * filt[(i+val/2-m)*(val*2) + (j+val/2-n)]);
       }

     }
   }

   //else{
    //if(type == 'e' or 's'){
    if (accumulation < 0) {
     accumulation = 0.0;
	}
    if (accumulation > 1) {
     accumulation = 1.0;
	}
    //}
    output[m*width + n] = accumulation;
   //}

   //printf("%f ", accumulation);
   }

//main program
int main(int argc, char **argv)
{
   //timer for the total program run time
   StopWatchInterface *timer1 = NULL;
   sdkCreateTimer(&timer1);
   sdkStartTimer(&timer1);

   // allocate the filter array
   int k_size = 3; //same for rows and cols
   //int k_size = 15; //same for rows and cols (CHANGE FILTER SIZE HERE)
   int kval = k_size;
   unsigned int fsize = k_size * k_size * sizeof(float);

   float *gauss = (float *) malloc(k_size * k_size * sizeof(float));
   float *sobel_x = (float *) malloc(k_size * k_size * sizeof(float));
   float *sobel_y = (float *) malloc(k_size * k_size * sizeof(float));
   //char f_typ = 's';

    // use to create filters
    createfilt2('s', gauss, k_size);
    createfilt2('x', sobel_x, k_size);
    createfilt2('y', sobel_y, k_size);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);


    //create output array
    float *img_gauss = (float *) malloc(size);
    float *img_gx = (float *) malloc(size);
    float *img_gy = (float *) malloc(size);
    float *img_gMag = (float *) malloc(size);
    float *img_gDir = (float *) malloc(size);
    float *img_nms = (float *) malloc(size);
    float *img_fOut = (float *) malloc(size);

    float *dData = NULL;
    // Allocate device memory and copy filt data
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    checkCudaErrors(cudaMemcpy(dData,
                               hData,
                               size,
                               cudaMemcpyHostToDevice));

    // Allocate device memory for filt
    float *img_out = NULL;
    // Allocate device memory and copy filt data
    checkCudaErrors(cudaMalloc((void **) &img_out, size));
    checkCudaErrors(cudaMemcpy(img_out,
                               hData,
                               size,
                               cudaMemcpyHostToDevice));


    //const unsigned int MAX_FILTER_SIZE = 79;
    //__device__ __constant__ float d_cFilterKernel[ MAX_FILTER_SIZE * MAX_FILTER_SIZE ];

    // Allocate constant memory for filt
    unsigned int filterKernelSizeByte = fsize;//filterSize * filterSize * sizeof(float);
    cudaMemcpyToSymbol( d_cFilterKernel, gauss, filterKernelSizeByte, 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( d_cFilterKernel1, sobel_x, filterKernelSizeByte, 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( d_cFilterKernel2, sobel_y, filterKernelSizeByte, 0, cudaMemcpyHostToDevice );



    dim3 dimBlock(64, 4, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    //dim3 dimGrid(256,1,1);//(32, 32, 1);//(width / dimBlock.x, height / dimBlock.y, 1);
    //dim3 dimBlock(1024, 1, 1);//(width / dimGrid.x, height / dimGrid.y, 1);

    // Warmup
    //transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t (sizeof(float)*64*64*64*64));

    // Execute the kernel
    conv_kern<<<dimGrid, dimBlock>>>(img_out, dData, height, width, kval, 'g');
    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    //sdkStopTimer(&timer);
    //printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    //printf("%.2f Mpixels/sec\n",


    //checkCudaErrors(cudaDeviceSynchronize());
    //StopWatchInterface *timer = NULL;
    //sdkCreateTimer(&timer);
    //sdkStartTimer(&timer);


    //do the convolution for gaussian smoothing
    //convolve( img_gauss, hData, gauss, height, width, k_size, 'g');


    // Allocate mem for the result on host side
    //float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(img_gauss,
                               img_out,
                               size,
                               cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaMemcpy(dData,
                               img_gauss,
                               size,
                               cudaMemcpyHostToDevice));

    conv_kern<<<dimGrid, dimBlock>>>(img_out, dData, height, width, kval, 'x');

    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(img_gx,
                               img_out,
                               size,
                               cudaMemcpyDeviceToHost));

    /*checkCudaErrors(cudaMemcpy(dData,
                               img_gauss,
                               size,
                               cudaMemcpyHostToDevice));*/

    conv_kern<<<dimGrid, dimBlock>>>(img_out, dData, height, width, kval, 'y');

    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(img_gy,
                               img_out,
                               size,
                               cudaMemcpyDeviceToHost));

    //do convolution for edge filtering in x and y, and  get magnitude and direction
    //convolve( img_gx, img_gauss, sobel_x, height, width, k_size, 'x');
    //convolve( img_gy, img_gauss, sobel_y, height, width, k_size, 'y');
    GetMag(img_gMag, img_gx, img_gy, height, width);
    //GetDir(img_gDir, img_gx, img_gy, height, width);
    NMSuppress(img_nms, img_gMag, img_gx, img_gy, 0.17, 0.199, width, height);
    Hysteresis(img_fOut, img_nms, img_gx, img_gy, 0.15, 0.199, width, height);
    sdkStopTimer(&timer);

    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    
   

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_Cgauss.pgm");
    sdkSavePGM(outputFilename, img_gauss, width, height);
    printf("Wrote '%s'\n", outputFilename);

    //char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_Cmag.pgm");
    sdkSavePGM(outputFilename, img_gMag, width, height);
    printf("Wrote '%s'\n", outputFilename);

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_Cnms.pgm");
    sdkSavePGM(outputFilename, img_nms, width, height);
    printf("Wrote '%s'\n", outputFilename);

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_Chys.pgm");
    sdkSavePGM(outputFilename, img_fOut, width, height);
    printf("Wrote '%s'\n", outputFilename);


   //free up created arrays
   free(gauss);
   free(sobel_x);
   free(sobel_y);

   free(img_gauss);
   free(img_gx);
   free(img_gy);
   free(img_gMag);
   free(img_gDir);
   free(img_nms);
   free(img_fOut);

    sdkStopTimer(&timer1);
    printf("Total time: %f (ms)\n", sdkGetTimerValue(&timer1));
    printf("Overhead time: %f (ms)\n", sdkGetTimerValue(&timer1)-sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer1);
    sdkDeleteTimer(&timer);
    //printf("Hello World!");
    return 0;
}

//functions

void createfilt2(char a, float* arr, int size){
if (a == 'x'){
  arr[0*size +0] = -1;
  arr[0*size +1] = 0;
  arr[0*size +2] = 1;
  arr[1*size +0] = -2;
  arr[1*size +1] = 0;
  arr[1*size +2] = 2;
  arr[2*size +0] = -1;
  arr[2*size +1] = 0;
  arr[2*size +2] = 1;  
}
else if(a == 'y'){
  arr[0*size +0] = -1;
  arr[0*size +1] = -2;
  arr[0*size +2] = -1;
  arr[1*size +0] = 0;
  arr[1*size +1] = 0;
  arr[1*size +2] = 0;
  arr[2*size +0] = 1;
  arr[2*size +1] = 2;
  arr[2*size +2] = 1;
}

else{
  arr[0*size +0] = 0.0625;
  arr[0*size +1] = 0.125;
  arr[0*size +2] = 0.0625;
  arr[1*size +0] = 0.125;
  arr[1*size +1] = 0.25;
  arr[1*size +2] = 0.125;
  arr[2*size +0] = 0.0625;
  arr[2*size +1] = 0.125;
  arr[2*size +2] = 0.0625;
}


}

/*void convolve(float* output, float* input, float* filt, int height, int width, int k_size, char type){

 for (int m = 0; m < height; m++ ) { //i
   for (int n = 0; n < width; n++ ) {//j
   float accumulation = 0;
   //float weightsum = 0;
   int val = k_size/2;
   for (int i = m-(val); i < (m+val+1); i++ ) {//k
     for (int j = n-(val); j < (n+val+1); j++ ) {//m
       //if((m+i >= 0 && m+i < height) && (n+j >=0 && n+j< width) ){
       if(i>=0 && i<width && j>=0 && j< height ){
       float k = input[(i)*height + (j)];//input(m+i, n+j); right by a b
       accumulation = accumulation + (k * filt[(i+val-m)*(k_size) + (j+val-n)]);//filt[(i+1)*(val) + (j+1)];//accumulation += k * kernel[1+i][1+j];
       //if (type == 'a'){
       //weightsum += filt[(i+val-1)*(val)+(j+val-1)]; //weightsum += kernel[1+i][1+j];shift is here
       //}
       }

     }
   }

    if (accumulation > 1.0) {
     accumulation = 1.0;
	}
    if (accumulation < 0.0) {
     accumulation = 0.0;
	}
    
    output[m*height + n] = accumulation;

   }
 }

}*/

void GetMag(float* output, float* imx, float* imy, int height, int width){
  for (int m = 0; m < height; m++ ) { //i
    for (int n = 0; n < width; n++ ) {//j
      float accumulation = sqrt(imx[m*height + n]*imx[m*height + n] + imy[m*height + n]*imy[m*height + n]); 
      output[m*height + n] = accumulation;

    }
  }
}

void GetDir(float* output, float* imx, float* imy, int height, int width){
  for (int m = 0; m < height; m++ ) {
    for (int n = 0; n < width; n++ ) {
      float theta = atan(imy[m*height + n]/imx[m*height + n]); 
      output[m*height + n] = theta;

    }
  }
}

void NMSuppress(float* output, float* G,float* after_Gx, float* after_Gy, float thr_min, float thr_max, int width, int height){

int nx = width;
int ny = height;
// Non-maximum suppression, straightforward implementation.
    for (int i = 1; i < nx - 1; i++){
        for (int j = 1; j < ny - 1; j++) {
            const int c = i + nx * j;
            const int nn = c - nx;
            const int ss = c + nx;
            const int ww = c + 1;
            const int ee = c - 1;
            const int nw = nn + 1;
            const int ne = nn - 1;
            const int sw = ss + 1;
            const int se = ss - 1;
 
            const float dir = (float)(fmod(atan2(after_Gy[c], after_Gx[c]) + M_PI, M_PI) / M_PI)*8;
 
            if (((dir <= 1 || dir > 7) && G[c] > G[ee] &&
                 G[c] > G[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && G[c] > G[nw] &&
                 G[c] > G[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && G[c] > G[nn] &&
                 G[c] > G[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && G[c] > G[ne] &&
                 G[c] > G[sw]))   // 135 deg
                output[c] = G[c];
            else
                output[c] = 0;
        }
     }

} 
void Hysteresis(float* output, float* nms,float* after_Gx, float* after_Gy, float thr_min, float thr_max, int width, int height){
    int nx = width;
    int ny = height;
    // Reuse array
    // used as a stack. nx*ny/2 elements should be enough.
    int *edges = (int*) after_Gy;
    memset(output, 0, sizeof(float) * nx * ny);
    memset(edges, 0, sizeof(float) * nx * ny);
 
    // Tracing edges with hysteresis . Non-recursive implementation.
    size_t c = 0;
    for (int j = 1; j < ny - 1; j++){
        for (int i = 1; i < nx - 1; i++) {
            if (nms[c] >= 0.199 && output[c] == 0.0) { // trace edges
    //printf("NOT HERE 3 \n");
                output[c] = 1.0;
                int nedges = 1;
                edges[0] = c;
 			    //printf("not here1");
                do {
                    nedges--;
                    const int t = edges[nedges];
 
                    int nbs[8]; // neighbours
                    nbs[0] = t - nx;     // n
                    nbs[1] = t + nx;     // s
                    nbs[2] = t + 1;      // w
                    nbs[3] = t - 1;      // e
                    nbs[4] = nbs[0] + 1; // nw
                    nbs[5] = nbs[0] - 1; // ne
                    nbs[6] = nbs[1] + 1; // sw
                    nbs[7] = nbs[1] - 1; // se
                
                    for (int k = 0; k < 8; k++)
                        if (nms[nbs[k]] >= 0.17 && output[nbs[k]] == 0.0) {
			    //printf("am here! ");
                            output[nbs[k]] = 1.0;
                            edges[nedges] = nbs[k];
                            nedges++;
                        }
                } while (nedges > 0);
            }
            c++;
        }
    }
    //free(after_Gx);
    //free(after_Gy);
    //free(G);
    //free(nms);



}

