// Includes, system
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



const char *imageFilename = "paper.pgm";

void serial(unsigned char *hdata,unsigned char *outdata, unsigned int width, unsigned int height,int window,float K){

for (int i=0; i < height ; i++){ 
	for (int j=0; j < width ;j++){
		float sum = 0;
		int square_sum = 0;
 		int index = i * width + j;
		int window_begin_x = ((0)>(i - window/2))?(0):(i - window/2);
 		int window_begin_y = ((0)>(j - window/2))?(0):(j - window/2);
		int window_end_x  = (((height-1)<(i + window/2))?(height-1):(i + window/2));
		int window_end_y  = (((width-1)<(j + window/2))?(width-1):(j + window/2));
		float window_size = (window_end_x - window_begin_x) * (window_end_y -window_begin_y );
 		for (int x= window_begin_x; x < window_end_x; x++){
 			for (int y= window_begin_y; y < window_end_y;y++){

				if (x*width+y <= width*height){ 				
				int temp = hdata[x*width+y];
				sum = sum + temp;
				square_sum = square_sum + temp * temp;
				}			
			}
		}
 		float mean = sum / window_size;
		float threshold = mean+K*sqrt((square_sum - mean*mean)/ window_size);
		if (threshold < hdata[index]){
		outdata[index] = 255;}
 		else{ 
		outdata[index] = 0;}
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
  unsigned char *hData = NULL; 
  //unsigned char *dOut = NULL;
  char outputFilename[1024];

  sdkLoadPGM(imagePath, &hData, &width, &height);
	

  unsigned int size = width * height * sizeof(unsigned char);
  unsigned char *hOutputData = (unsigned char *)malloc(size);

  StopWatchInterface *timer1 = NULL;
  sdkCreateTimer(&timer1);
  sdkStartTimer(&timer1);
  serial(hData, hOutputData, width, height, 80,-0.2);
  sdkStopTimer(&timer1);

 // cudaMemcpy(hOutputData, dOut, size, cudaMemcpyDeviceToHost);

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out1S.pgm");
  sdkSavePGM(outputFilename, hOutputData, width, height);
  sdkStopTimer(&timer);

    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer1));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer1) / 1000.0f)) / 1e6);
    printf("Overhead TIme: %f (ms) \n",sdkGetTimerValue(&timer)-(sdkGetTimerValue(&timer1)));
    printf("TOTAL TIME: %f (ms)\n ",(sdkGetTimerValue(&timer)));
  sdkDeleteTimer(&timer);
  sdkDeleteTimer(&timer1);
  return 0;



}
