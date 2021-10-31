#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "time.h"


// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define BLOCK_SIZE 2
#define WINDOW_SIZE 20
const char *imageFilename = "paper.pgm";

__global__ void serial(float *hdata,float *outdata, unsigned int width, unsigned int height,float K){

	__shared__ float Shared[BLOCK_SIZE+WINDOW_SIZE-1][BLOCK_SIZE+WINDOW_SIZE-1];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = (blockIdx.y * BLOCK_SIZE ) + ty;
	int col_o = (blockIdx.x * BLOCK_SIZE ) + tx;
 	int row_i = row_o - WINDOW_SIZE/2;
	int col_i = col_o - WINDOW_SIZE/2;
 	int index = row_o * width + col_o;
	if (row_i >=0 && row_i <height && col_i >=0 && col_i < width){
		Shared[ty][tx] = hdata[row_i*width+col_i];
	} 
	else{
		Shared[ty][tx] = 0.0f;
	}
 	__syncthreads();

	
	 int  window_begin_x = ( row_o < WINDOW_SIZE/2) ? 0 : row_o - WINDOW_SIZE/2;
	 int  window_begin_y = ( col_o < WINDOW_SIZE/2) ? 0 : col_o - WINDOW_SIZE/2;
	 int  window_end_x = ( height-1 < row_o + WINDOW_SIZE/2) ? height-1 : row_o + WINDOW_SIZE/2;
	 int window_end_y = ( width-1 < col_o + WINDOW_SIZE/2) ? width-1 : col_o + WINDOW_SIZE/2;

 	float window_size = (window_end_x - window_begin_x+1) * (window_end_y - window_begin_y+1 );

	float threshold;
	if (ty < BLOCK_SIZE && tx < BLOCK_SIZE){
 		float sum = 0;
		float square_sum = 0;
 	for(int i= 0; i< WINDOW_SIZE;i++){
		for(int j= 0; j< WINDOW_SIZE;j++){
			float temp = Shared [(i+ty)][(j+tx)];
			sum =  sum + temp;
			square_sum = square_sum + temp * temp;
		}
	}

	float mean = sum / window_size;
	threshold = mean+K*sqrtf((square_sum - mean*mean)/ window_size);

	}

	if (row_o < height && col_o < width )
	if (ty < BLOCK_SIZE && tx < BLOCK_SIZE){
		if (threshold < hdata[index]){
			outdata[index] = 1;
		}else{
			outdata[index] = 0;
		}
	}
	
}


int main(int argc, char **argv){
  //timer for entire program 
  StopWatchInterface *timer =NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  //loading the image
  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL){
      printf("Unable to source image file: %s\n", imageFilename);
      exit(EXIT_FAILURE);
  }

  unsigned int width, height;
  float *hData = NULL; 
  //unsigned char *dOut = NULL;
  char outputFilename[1024];

  sdkLoadPGM(imagePath, &hData, &width, &height);
	

  unsigned int size = width * height * sizeof(float);
  float *hOutputData = (float *)malloc(size);

  

	float * DeviceInputData;
	float * DeviceOutputData;

	checkCudaErrors(cudaMalloc((void **) &DeviceInputData, size));
	checkCudaErrors(cudaMalloc((void **) &DeviceOutputData, size));

	cudaMemcpy(DeviceInputData,hData,size,cudaMemcpyHostToDevice);

	dim3 dimGrid, dimBlock;

	dimBlock.x = BLOCK_SIZE+WINDOW_SIZE-1 ;
	dimBlock.y = BLOCK_SIZE+WINDOW_SIZE-1 ;
	dimBlock.z = 1 ;
	
 	dimGrid.x = (width+BLOCK_SIZE-1)/BLOCK_SIZE;
	dimGrid.y = (height+BLOCK_SIZE-1)/BLOCK_SIZE;
	dimGrid.z = 1;


  StopWatchInterface *timer1 = NULL;
  sdkCreateTimer(&timer1);
  sdkStartTimer(&timer1);
  


  serial<<<dimGrid,dimBlock>>>(DeviceInputData, DeviceOutputData, width, height,-0.2);
  

  sdkStopTimer(&timer1);

 // cudaMemcpy(hOutputData, dOut, size, cudaMemcpyDeviceToHost);


checkCudaErrors(cudaMemcpy(hOutputData,DeviceOutputData,size,cudaMemcpyDeviceToHost));

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out1CUDA.pgm");

	
  sdkSavePGM(outputFilename, hOutputData, width, height);
  sdkStopTimer(&timer);

    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer1));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer1) / 1000.0f)) / 1e6);
    printf("Overhead TIme: %f (ms) \n",sdkGetTimerValue(&timer)-(sdkGetTimerValue(&timer1)));
	 printf("TOTAL TIME: %f (ms)\n",(sdkGetTimerValue(&timer)));
   sdkDeleteTimer(&timer);

  sdkDeleteTimer(&timer);
  sdkDeleteTimer(&timer1);
  return 0;



}
