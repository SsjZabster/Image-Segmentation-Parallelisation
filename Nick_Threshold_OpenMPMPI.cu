#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "time.h"
#include "mpi.h"

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))



const char *imageFilename = "NewsPaper1080.pgm";

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
  


    //MPI world topology 
    int process_id, num_processes,Quarter;
    int window = 80; 
    int k = -0.2;

	/* Find current task id */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	/* MPI status */
    MPI_Status status;
    MPI_Request send_north_req;
    MPI_Request send_south_req;
    MPI_Request send_west_req;
    MPI_Request send_east_req;
    MPI_Request recv_north_req;
    MPI_Request recv_south_req;
    MPI_Request recv_west_req;
    MPI_Request recv_east_req;
	
   

    if(process_id == 0){
    if((width * height)/4 % 2 == 0){

    	Quarter = (width * height)/4;
    	
    } 
    else{

	Quarter = ((width * height)+1)/4;  
	
    }
	
    }

	MPI_Bcast(&Quarter, 1, MPI_INT, 0, MPI_COMM_WORLD);


    	MPI_Barrier(MPI_COMM_WORLD);
	//printf("%i \n",Quarter);
 	//printf("%i \n",process_id);
	sdkStartTimer(&timer1);	
	if (process_id == 0) {
		unsigned char * firstHalf = (unsigned char *)malloc(Quarter*sizeof(unsigned char));	
		unsigned char * recv = (unsigned char *)malloc(Quarter*sizeof(unsigned char));
		memcpy(firstHalf, hData, Quarter*sizeof(unsigned char));
		serial(firstHalf,recv,width/2,height/2, window, k);
		memcpy(hOutputData, recv, Quarter*sizeof(unsigned char));
		
		
		//RECIEVE ALL THE HOUTPUTDATA
		
		MPI_Irecv(recv, Quarter, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD, &send_west_req);
		MPI_Wait(&send_west_req, &status);
		memcpy(&hOutputData[Quarter],recv,Quarter*sizeof(unsigned char));
		
		MPI_Irecv(recv, Quarter, MPI_UNSIGNED_CHAR, 2, 0, MPI_COMM_WORLD, &recv_south_req);
		MPI_Wait(&recv_south_req, &status);
		memcpy(&hOutputData[2*Quarter],recv,Quarter*sizeof(unsigned char));
		printf("%i  dfawfa  02200 \n",Quarter);
		MPI_Irecv(recv, Quarter, MPI_UNSIGNED_CHAR, 3, 0, MPI_COMM_WORLD, &recv_east_req);
		MPI_Wait(&recv_east_req, &status);
		memcpy(&(hOutputData[3*Quarter]),recv,Quarter*sizeof(unsigned char));
		printf("%i  dfawfa  03300 \n",Quarter);
	}

	else if (process_id == 1) {
		printf("%i  dfawfa  0100 \n",Quarter);
		unsigned char * secondhalf = (unsigned char *)malloc(Quarter*sizeof(unsigned char));
		unsigned char * recv = (unsigned char *)malloc(Quarter*sizeof(unsigned char));
		memcpy(secondhalf, &hData[Quarter], Quarter*sizeof(unsigned char));
		serial(secondhalf,recv,width/2,height/2, window, k);		
		MPI_Isend(recv, Quarter, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &send_west_req);
	}
	else if (process_id == 2) {
		printf("%i  dfawfa  0200 \n",Quarter);
		unsigned char * secondhalf = (unsigned char *)malloc(Quarter*sizeof(unsigned char));
		unsigned char * recv = (unsigned char *)malloc(Quarter*sizeof(unsigned char));
		memcpy(secondhalf, &hData[2*Quarter], Quarter*sizeof(unsigned char));
		serial(secondhalf,recv,width/2,height/2, window, k);	
		MPI_Isend(recv, Quarter, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &recv_south_req);

		}
	else if (process_id == 3) {
			printf("%i  dfawfa  0300 \n",Quarter);
		unsigned char * secondhalf = (unsigned char *)malloc(Quarter*sizeof(unsigned char));
		unsigned char * recv = (unsigned char *)malloc(Quarter*sizeof(unsigned char));
		memcpy(secondhalf, &hData[3*Quarter], Quarter*sizeof(unsigned char));
		serial(secondhalf,recv,width/2,height/2, window, k);
		MPI_Isend(recv, Quarter, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &recv_east_req);

	}
	

MPI_Barrier(MPI_COMM_WORLD);
	sdkStopTimer(&timer1);

if(process_id == 0){

 

 // cudaMemcpy(hOutputData, dOut, size, cudaMemcpyDeviceToHost);

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out1MPI.pgm");
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
	
}
MPI_Finalize();
return 0;
}
