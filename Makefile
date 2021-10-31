
all: kmeans_sr kmeans_cuda kmeans_mpi NTS NTSC NTSO CSER CCUDA COPEN

kmeans_sr: kmeans_sr.cu
	nvcc -I inc/ kmeans_sr.cu -o km_sr

kmeans_cuda:
	nvcc -I inc/ kmeans_cuda.cu -o km_cuda

kmeans_mpi: kmeans_mpi.cu
	nvcc -I inc -I /usr/lib/x86_64-linux-gnu/openmpi/include -L /usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi kmeans_mpi.cu -o km_mpi

NTS: Nick_Threshold.cu
	nvcc Nick_Threshold.cu -I inc -o NTS

NTSC: Nick_Threshold_Shared.cu
	nvcc Nick_Threshold_Shared.cu -I inc -o NTSC

NTSO: Nick_Threshold_OpenMPMPI.cu
	nvcc Nick_Threshold_OpenMPMPI.cu -I inc -I /usr/lib/x86_64-linux-gnu/openmpi/include -L /usr/lib/x86_64-linux-gnu/openmpi/lib -w -o NTSO

CSER: canny_sr.cu
	nvcc canny_sr.cu -I inc -o CSER

CCUDA: canny_cuda.cu
	nvcc canny_cuda.cu -I inc -o CCUDA

COPEN: Canny_openMPI.cu
	nvcc Canny_openMPI.cu -I inc -w -o COPEN

clean:
	rm km_sr km_cuda km_mpi NTS NTSC NTSO CSER CCUDA 
