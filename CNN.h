/* Jack Newman.
*/

/*=================================================================================================*/
/*Externeal Libaries*/
  #include <windows.h>
  #include <stdio.h>
	#include <stdlib.h>
	#include <string.h>
	#include <stdint.h>
	#include <time.h>
	#include <unistd.h>
  #include <time.h>
	#include <omp.h>
	#include <stdalign.h>
	#include <stdatomic.h>
	#include <math.h>
/*=================================================================================================*/
/*@D Dymmnically defined into statics by Pre_Processer: Template is ffor mnsit train*/
 // -F = Flag passed in. | -G = Generated 
 //1. Data Configuration: How the file is set up. 
		#define DTP 1        	  // -F 1=double, 0=short
		#define X 28          	// -F Input image width
		#define Y 28          	// -F Input image height
		#define OPLEN 10      	// -F Num outputs the network will have
		#define IOLEN X*Y   	 	// -G Num inputs flattened + 1 spot for type of output
		#define ROWS 60000      // -F DataSet Size
		#define COLS IOLEN+1    // -G Amount of cols per row. 
		#define FNAME "Locat"   // -F (DTP inserted directly),
	//2. Computional Configurations. 
		#define TRV 1           // -F  added 0 for silent mode, for hyperparamter tuning. 
		#define DTV 1           // -F  Controls how much you see training DT set up.
		#define SIMDID 256			// -F  
		#define IO_THRDS (int)(ROWS*.001) //-F Amount of threads.
		#define CP_THRDS 1
		#define GPUON 0
	/*@TRAIN INFO Taken directly from my old Nueral net projects*/
		#define Rand       	0
		#define initWE     	0.1   	   // -w <DOUBLE>   	Weight initialization parameter; default 0.1 
		#define alpha      	0.01     	 // -a <DOUBLE>		Learning rate for gradient descent; default 0.01
		#define lambda     	0          // -l <DOUBLE>		Regularization parameter; default 0.0
		#define epochLimit 	1000       // -e <INTEGER>	Epoch limit for gradient descent; default 1000: 
		#define btchSz     	1          // -m <INTEGER>	Batch Sz; 0 for full batch; default 1
	  #define batchStyle  1					 // 0=no preference, 1=uni-class, 2=2 classes per batch. 
		#define even_dis    0          // if batchStyle > 1, evenly distrbute combinations, else rnadom combinations.  
		#define TRNSZ .8        // -F  size of training data from given set. 

	//@E
	//In the future, batchStyle 2+ will either create a  can we come up with even-Distrubtion methods. 
	/*@Unique Foward/Back methods + activation Defines

	//@E
/*=================================================================================================*/
/*Dynamically Generated Items based off pre-porcessing choices*/
	#define Train_Sz (int)(ROWS*TRNSZ)
	#if DTP             
			#define Numeric double
			#define NScan "lf"
			#define NPrint "f"
			#define ScanN atof
			#define StrToN strtod
			#define NMAX __DBL_MAX__
			#define NMIN __DBL_MIN__
		#else
			#define Numeric float  
			#define NScan "f"
			#define NPrint "f"
			#define ScanN (float)atof
			#define StrToN strtof
			#define NMAX __FLT_MAX__
			#define NMIN __FLT_MIN__
	#endif

/*=================================================================================================*/
/*Vector Helpers and Marcos, pre-generated basedd off choice*/
	#include <SIMD_MARCO.h>
	#define VECT_I_ALIGN (_VECT_WIDTH / VECT_INT_COUNT) // 16 / 32 / 64 depending on SSE/AVX/AVX512
	#define PADDED_IVECT (((OPLEN + VECT_INT_COUNT - 1) / VECT_INT_COUNT) * VECT_INT_COUNT)

/*=================================================================================================*/
/* Structures */
	typedef void (*FunctionPointer)();

/*=================================================================================================*/
/*  Global Variables */
		//uint64_t s0 = 0x123456789abcdefULL,  0xfedcba987654321ULL; // 
		_Thread_local uint64_t s0, s1; //for now we will incur a measily cost of this onto each of our threads. 
	
		alignas(_VECT_WIDTH/VECT_NUMERIC_COUNT) Numeric IO_Layer[ROWS*COLS]; //Padded for vectorized operations. 
		//For pure optimization, we could have option to minimax scale before hand, and make this const. Either way, if we have to pre-process


		Numeric Row_Access[ROWS]; //Indices of the start each row. If R and batchStyle = 1, then they will be sorted by classfier. 
		int * trn_btchs;  
		int total_btchs = 0;

		
		#if batchStyle > 0
			alignas(VECT_I_ALIGN) int Frequency[PADDED_IVECT];
			atomic_int boundary_ptr[OPLEN];  //This representss, atomically, where each class is withn their boundary, 
			int class_boundaries[OPLEN+1]; 	 //Numerical list of class boundaries. 
		#else
			int *Frequency;						
			atomic_int *boundary_ptr;
			int class_boundaries[OPLEN];	  //GETS repurposed! So we can evenely threed our sorting! 
		#endif

/*=================================================================================================*/
/*  MARCOS */   
		#define E_MAC(msg) {perror(msg); return;}
		#define ST(a,i,j) a[(i)*(COLS) + (j)]
		#define DEFINE_LAYER(NAME, FORWARD, ACT) \
			void NAME(float *in, float *out, int len) { \
					FORWARD(in, out, len, ACT); \
			}

/*=================================================================================================*/
/* Data_set Handeling and CNN set up function declarations */

	/*=========================================================================*/
	/*Main Functions*/
		int main(int argc, char * argv[]);
		void train();
	/*=========================================================================*/
	//0. Data Processing OPTIMIZED and profiled. 
		void initilize_DT();
		void create_IO_Layer();
		void minMax_Scale(); 
		 //Apply min-max normalization to scale feature values to the range [-1, 1]. For each feature j, 
		 //use its min (Lj) and max (Uj) in the training set to transform the values: x'_ij = -1 + 2 * ((x_ij - Lj) / (Uj - Lj))
	/*=========================================================================*/
	// 1. THREADED SORT BY CLASSIFIER Needs heavy reworking. 
		void efficent_freq();	
			// Find class boundaries in sorted array
		void threaded_atomic_ptr_sort(); 
				//Threaded sort implementation using counting sort approach
	/*=========================================================================*/
	// 2. Creating Mini-Batches
		void create_mini_batches();
		int set_class_size(int btch_sz);
	/*=========================================================================*/
	// 4. RANDOMIZATION METHOD
	  void init_prng_threads();
		void shuffle_mb_postions();
		void shuffle_PT_in_bounds(); 
		void shuffle_PT();
		int xoroshiro128_next_range(int min, int max);
		uint64_t xoroshiro128_next();
		

		
	
/*=================================================================================================*/
/* CNN Functions */
	static inline float relu(float x) { return (x > 0) ? x : 0; }
	static inline float softmax(float x) { return x; } // placeholder

	void convolution_forward(float *in, float *out, int len, float (*act)(float)) {
			for (int i = 0; i < len; i++) {
					float sum = in[i]; // fake convolution
					out[i] = act(sum);
			}
	}

	void maxpool_forward(float *in, float *out, int len, float (*act)(float)) {
			for (int i = 0; i < len; i++) {
					float val = in[i]; // fake pooling
					out[i] = act(val);
			}
	}

	/*=========================================================================*/
	//5. UTILITY FUNCTIONS
		void timer(FunctionPointer func, const char *msg);
		void train_time();