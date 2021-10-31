#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
// Include extra helper functions for CUDA - handles image load/save
//#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>       

const int k = 2;

const char *imageFilename = "suburb.ppm";


void kmeans(unsigned char *g_idata,unsigned char *g_odata, unsigned int width, unsigned int height,float * clus,int start,int finish,float* cTemp,float* count){

  float newsum, r, g, b =0.0;
  float sum=999999.0;
  int cVal=0;
  unsigned int size = k;
  for(int i=start;i<finish;i++){
    for(int j=0;j<width;j++){

      newsum=0.0;
      r=0.0;
      g=0.0;
      b=0.0;
      sum=999999.0;
      cVal=0;

      for(int a=0;a<size;a++){

        newsum=(abs((clus[a*3])-(g_idata[i*width*4+j*4]))+abs(clus[a*3+1]-(g_idata[i*width*4+j*4+1]))+abs(clus[a*3+2]-(g_idata[i*width*4+j*4+2])));
        if(sum>newsum){
          cVal=a;
          sum=newsum;
          r=(g_idata[i*width*4+j*4]);
          g=(g_idata[i*width*4+j*4+1]);
          b=(g_idata[i*width*4+j*4+2]);
          g_odata[i*width*4+j*4]=cVal;
          g_odata[i*width*4+j*4+1]=cVal;
          g_odata[i*width*4+j*4+2]=cVal;
        }
      }


      count[cVal]+=1.0;
      cTemp[cVal*3]+=r;
      cTemp[cVal*3+1]+=g;
      cTemp[cVal*3+2]+=b;
 }
}

}



void clustering(unsigned char *in_data,unsigned char *out_data, unsigned int width, unsigned int height,float * clus){


int stop=0;
  float*  cTemp = (float *)malloc(k*3* sizeof(float));// pass through , dont declare locally
  float*  count = (float *)malloc(k* sizeof(float));
  float*  cTempS = (float *)malloc(k*3* sizeof(float));// pass through , dont declare locally
  float*  countS = (float *)malloc(k* sizeof(float));
  unsigned int size = k;
  int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for(int i=0;i<size;i++){
    cTempS[i*3]=0;
    cTempS[i*3+1]=0;
    cTempS[i*3+2]=0;
    countS[i]=0;
  }
  int num=0;
  MPI_Comm_size(MPI_COMM_WORLD, &num);
  while(stop==0){
    for(int i=0;i<size;i++){
      cTemp[i*3]=0;
      cTemp[i*3+1]=0;
      cTemp[i*3+2]=0;
      count[i]=0;
    }
      if(rank==num-1){
        kmeans(in_data, out_data, width, height, clus,floor(height/(num-1))*rank,height,cTemp,count);
      }else{
        kmeans(in_data, out_data, width, height, clus,floor(height/(num-1))*rank,floor(height/(num-1))*(rank+1),cTemp,count);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allreduce(cTemp, cTempS, k*3 ,  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(count, countS, k ,  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      if(rank==0){
        float error=0.0;
        for(int a=0;a<size;a++){
          if(countS[a]!=0.0){
            error=error+abs(clus[a*3]-(cTempS[a*3]/countS[a]))+abs(clus[a*3+1]-(cTempS[a*3+1]/countS[a]))+abs(clus[a*3+2]-(cTempS[a*3+2]/countS[a]));
          }else{
            printf("%d\n",size);
          }
        }
        if(error/(float)size<10){
          stop=1;
        //  MPI_Bcast(&stop,1,MPI_INT,0,MPI_COMM_WORLD);
          printf("%d\n",size );
        }
        for(int s=0;s<size;s++){
          if(countS[s]!=0){
          clus[s*3]=cTempS[s*3]/countS[s];
          clus[s*3+1]=cTempS[s*3+1]/countS[s];
          clus[s*3+2]=cTempS[s*3+2]/countS[s];
         }


        }
      }
      MPI_Bcast(&stop,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(clus,k*3,MPI_FLOAT,0,MPI_COMM_WORLD);
      }
      unsigned int size1 = width * height*4;
      unsigned char * out_dataf = (unsigned char *)malloc(size1*sizeof(unsigned char));
      MPI_Allreduce(out_data,out_dataf,size1,MPI_UNSIGNED_CHAR,MPI_SUM,MPI_COMM_WORLD);



  //}
 if(rank==0){
   for(int i=0;i<height;i++){
     for(int j=0;j<width;j++){
       out_data[i*width*4+j*4]=clus[out_dataf[i*width*4+j*4]*3];
       out_data[i*width*4+j*4+1]=clus[out_dataf[i*width*4+j*4+1]*3+1];
       out_data[i*width*4+j*4+2]=clus[out_dataf[i*width*4+j*4+2]*3+2];
     }
   }
 }

}





int main(int argc, char **argv){
  //timer for entire program 
  /*StopWatchInterface *timer =NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);*/

  //StopWatchInterface *timer1 = NULL;
  double start,finish,proS,proF;
  start=MPI_Wtime();

  MPI_Init(&argc, &argv);

  //loading the image
  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL){
      printf("Unable to source image file: %s\n", imageFilename);
      exit(EXIT_FAILURE);
  }
  int rank;
  unsigned int width, height;
  unsigned char *hData = NULL;
  //unsigned char *dOut = NULL;
  char outputFilename[1024];

  sdkLoadPPM4ub(imagePath, &hData, &width, &height);
  //width = 702;
  //height = 336;


  //write image to file
  //FILE *f = fopen("myimg.data", "wb");
  //fwrite(hData, sizeof(float), width*height, f);
  //fclose(f);

  //read in image
  //FILE *imf = fopen("suburb.data", "rb");
  //fread(OutData, sizeof(float), width*height, imf);


  int csize = k;
  unsigned int size = width * height * sizeof(unsigned char)*4;
  unsigned char * hOutputData = (unsigned char *)malloc(size);

  //array 4 cluster centres on host and 4 update
  float* clusterH = (float *)malloc(csize*3* sizeof(float));
  //float* clusterT = (float *)malloc(csize*3* sizeof(float));
  float* counter = (float *)malloc(csize* sizeof(float));


  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0){
    int randomn;// randomnumber1, randomnumber2;
    for(int i=0;i<csize;i++){
      randomn = rand() % height*width;
      //randomnumber1 = rand() % height*width;
      //randomnumber2 = rand() % height*width;

      clusterH[i*3]=hData[randomn*4];
      clusterH[i*3+1]=hData[randomn*4+1];
      clusterH[i*3+2]=hData[randomn*4+2];
      /*cluster[i*3]=hData[randomnumber*1];
      cluster[i*3+1]=hData[randomnumber1*2+1];
      cluster[i*3+2]=hData[randomnumber2*3+2];*/
    }
}

  MPI_Bcast(clusterH,k*3,MPI_FLOAT,0,MPI_COMM_WORLD);
  proS=MPI_Wtime();
  clustering(hData, hOutputData, width, height, clusterH);
  proF=MPI_Wtime()-proS;

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_outmpi.ppm");
  if(rank==0){
  sdkSavePPM4ub(outputFilename, hOutputData, width, height);
   }
  finish=MPI_Wtime()-start;

  printf("Processing time: %f (ms)\n", proF*1000);
  printf("%.2f Mpixels/sec\n", ((width *height) / (proF*1000/1000.0f ) )/ 1e6);
  printf("Overhead time: %f (ms)\n",(finish-proF)*1000);
   printf("Total time: %f (ms)\n", finish*1000);
  MPI_Finalize();//put at end?
  //printf("Overhead time: %f (ms)\n",(finish-proF)*1000);
  //sdkDeleteTimer(&timer1);
  return 0;

}
