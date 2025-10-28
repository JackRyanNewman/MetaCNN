/*┌─────────────────────────────────────────────────────────────────────────────┐
  │ CNN                                                                         │                                                                             │
  │   • © 2025 Jack Ryan Newman:                                                │
  ├─────────────────────────────────────────────────────────────────────────────┤
  └─────────────────────────────────────────────────────────────────────────────┘*/
/*=================================================================================================*/
/*Dymanic Setting*/
	#include <stdint.h>
	#if PROCESS_GUARD
		#include <CNN_c.h>
	#else
		#include <CNN.h>
	#endif

	//Change all post to prencrments...

/*=================================================================================================*/
int main(int argc, char * argv[]) {
	/*=========================================================================*/
		/* 1. parse inputs and valadition */
		///omp_set_num_threads(60);
		putenv("OMP_STACKSIZE=128k");

		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
		timer(train, "Run All");
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
		printf("CURRENTLY: python ONLY READING FILE+MINIMAX+FREQUNECY CONUT+RANDOMIZED DATASET 50x\n");
		printf("CURRENTLY: C READING FILE+MINIMAX+FREQUNECY CONUT+THREADESORT+MINIBATCHES+CREATES RANDOMIXZE UNI SINGLE BATCH CLASSSES+50x full set randomizations!");
		return 0;
}
/*=================================================================================================*/
//Train Level: Currentlty desingated to test functions, and speed them up. Everything after 1. Was 
//Created in scrum, just to get some ideas, they are not optimnal. I dont like that. 
 void train(){
		create_IO_Layer();	
		minMax_Scale();
		if(batchStyle > 0){
				efficent_freq();
				threaded_atomic_ptr_sort();
				init_prng_threads();
		}
		create_mini_batches();
		for(int i=0; i < 100*100; i++) {
			if(batchStyle>0) shuffle_PT_in_bounds();
			else shuffle_PT();
		}
 }

 void train_time(){
		timer(create_IO_Layer, "create_IO_Layer");
		timer(minMax_Scale, "minMax_Scale");

		if(batchStyle > 0){
				timer(efficent_freq, "efficent_freq");
				timer(threaded_atomic_ptr_sort, "threaded_atomic_ptr_sort");
				timer(init_prng_threads, "init_prng_threads");
		}

		timer(create_mini_batches, "create_mini_batches");
		timer(shuffle_PT_in_bounds, "Single Random");

 }


/*=================================================================================================*/
//0. Data Intiliziation: Loading, Regularization, and Pre-Sorting. 
 	void initilize_DT(){
	}

	void create_IO_Layer(){
		if (TRV != 0) printf("* Reading %s\n", FNAME);
		
		#pragma omp parallel for simd 	//Pre-Fetch memory.
		for (size_t i =0; i != ROWS*COLS; ++i) IO_Layer[i] = 0;
		
		
 		HANDLE hFile = CreateFileA(FNAME, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (hFile == INVALID_HANDLE_VALUE) E_MAC("Failed to open file");
 
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        E_MAC("Failed to get file size");
    }
    OVERLAPPED overlapped = {0}; DWORD bytesRead;
		size_t expectedSize = ROWS * COLS * sizeof(Numeric);
    overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    BOOL success = ReadFile(hFile, IO_Layer, expectedSize, &bytesRead, &overlapped);
    if (!success && GetLastError() == ERROR_IO_PENDING) {
        // Wait for async I/O to complete
        success = GetOverlappedResult(hFile, &overlapped, &bytesRead, TRUE);
    }
    if (!success || bytesRead != expectedSize) {
        CloseHandle(overlapped.hEvent); CloseHandle(hFile);
        E_MAC("Failed to read file asynchronously");
    }

    CloseHandle(overlapped.hEvent);
    CloseHandle(hFile);
		if(Rand){
			Numeric* temp_row = malloc(COLS * sizeof(Numeric));
			for (int i = ROWS - 1; i > 0; i--) { //Vectorized these swaps. 
					int j = rand() % (i + 1);
					memcpy(temp_row, &IO_Layer[i * COLS], COLS * sizeof(Numeric));
					memcpy(&IO_Layer[i * COLS], &IO_Layer[j * COLS], COLS * sizeof(Numeric));
					memcpy(&IO_Layer[j * COLS], temp_row, COLS * sizeof(Numeric));
			}
			free(temp_row);
		}

  }
	
	void minMax_Scale(){
		Numeric MinMax[IOLEN][2];			
		
		#pragma omp parallel for
		for (int col = 0; col < IOLEN; col++) {
			Numeric currMin = NMAX, currMax = NMIN;
			#pragma omp simd
			for (int row = 0; row < Train_Sz; row++) {
					Numeric tmp = IO_Layer[row * IOLEN + col]; // Row-major access
					if (tmp < currMin) currMin = tmp;
					if (tmp > currMax) currMax = tmp;
			}
			// Store results after loop to reduce memory writes
			MinMax[col][0] = currMin;
			MinMax[col][1] = currMax;
		}
	
		if(TRV!=0) {
			printf("* Doing train/validation split\n"
						"* Scaling features\n");
		}
		if(TRV>1) {
			printf("  * min/max values on training set");
			int count;
			for(int i=0; i < IOLEN; i++) {
				printf("      Feature %d: %.4f" NPrint ", %.4" NPrint "%n", i + 1, MinMax[i][0], MinMax[i][1], &count);
			}
		}
		
		//Making Affine Transofmration: Of the equation. 
		//So we need to. Precompute scale and bias for each column to unify the logic
		Numeric scale[IOLEN], bias[IOLEN];
		for (int c = 0; c < IOLEN; c++) {
				Numeric min = MinMax[c][0];
				Numeric div = MinMax[c][1]- min;
				if (div == 0) {
						scale[c] = 0;
						bias[c] = -1;
				} else {
						scale[c] = 2 / div;
						bias[c] = -1 - scale[c] * min;
				}
		}

		// Now apply the affine transformation with threading over rows and vectorization over columns
		#pragma omp parallel for //I will let it handle the thread amount. 
		for (int row = 0; row < ROWS; row++) {
				#pragma omp simd 
				for (int c = 0; c < IOLEN; c++) {
						ST(IO_Layer, row, c) = scale[c] * ST(IO_Layer, row, c) + bias[c];
				}
		}
		
	}
/*=================================================================================================*/
//3. Uni-class creation.
	
	void efficent_freq(){
		if (TRV > 5) printf("* Starting efficent Frequency %s\n", FNAME);
		#pragma omp parallel num_threads(IO_THRDS)
		{
			int tid   = omp_get_thread_num();
			alignas(VECT_I_ALIGN) int local[PADDED_IVECT] = {0};  // private per thread
			int start = tid * (Train_Sz / IO_THRDS) + (tid < Train_Sz % IO_THRDS ? tid : Train_Sz % IO_THRDS);
			int end   = start + (Train_Sz / IO_THRDS) + (tid < Train_Sz % IO_THRDS ? 1 : 0);
			
			for (int r = start; r < end; r++) local[(int)IO_Layer[r * COLS + (COLS - 1)]]++;

			// Merge into global Frequency
			#pragma omp critical
			{ for (int vec_pos = 0; vec_pos < PADDED_IVECT; vec_pos += VECT_INT_COUNT) {
					VECT_INT v_local = VECT_LOAD_INT((VECT_INT*)&local[vec_pos]);
					VECT_INT v_freq  = VECT_LOAD_INT((VECT_INT*)&Frequency[vec_pos]);
					v_freq = VECT_ADD_INT(v_freq, v_local);
					VECT_STORE_INT((VECT_INT*)&Frequency[vec_pos], v_freq);
				}
			}
		}
	
	}

	void threaded_atomic_ptr_sort(){
			// Step 1: Create cumulative frequency array for class boundarie
			class_boundaries[0] = 0;
			for (int i = 0; i < OPLEN; i++){
				class_boundaries[i+1] = class_boundaries[i]+Frequency[i];
				atomic_store(&boundary_ptr[i], class_boundaries[i]);
			}

			#pragma omp parallel num_threads((int)OPLEN/2)
			{
				int tid = omp_get_thread_num();
				int start = tid * (Train_Sz / IO_THRDS) + (tid < Train_Sz % IO_THRDS ? tid : Train_Sz % IO_THRDS);
    		int end = start + (Train_Sz / IO_THRDS) + (tid < Train_Sz % IO_THRDS ? 1 : 0);
				for(int r = start; r < end; r++){ 
					 int CLS = IO_Layer[r * COLS + (COLS - 1)];						//Get us the classfier type. 
					 int pos = atomic_fetch_add(&boundary_ptr[CLS], 1); 	//Then move atomtic along boundary. 
					 Row_Access[pos] = IO_Layer[r * IOLEN ];							//Point sorted data to nenw postion. 
				}					
			}			
			
	}

// /*=================================================================================================*/
//2. Creating Mini-Batches

	void create_mini_batches(){
		//BatchSize is a perctange of trainSize if < 1, else if its >= it becomees trainSz, else its just specifed num
		int btch_sz = btchSz < 1 ? (int)(Train_Sz * btchSz): btchSz >= Train_Sz? Train_Sz: btchSz;
		if(batchStyle != 0 && ! btch_sz%OPLEN) btch_sz = btch_sz%OPLEN; //Get remainder that needed
		if(btch_sz >= Train_Sz) btch_sz = Train_Sz;											//Check again if remainder execes limit. 
		if(DTV>0) printf("Current BtchSz = %d\n", btch_sz);


		if(btch_sz==Train_Sz || batchStyle==0){ //If full batch, size = 2, else if there is a remainder add +2, *2 for start and stops. 		
			int arrSz = Train_Sz==btch_sz? 2: ((int)(Train_Sz/btch_sz))*2;
			trn_btchs = malloc(arrSz*sizeof(int));		//Add a start and stop for each postion!
			int b = 0, loc = 0;
			for(; b< (arrSz/2)-1; b++, loc+=btch_sz){
				trn_btchs[b*2]   = loc; 
				trn_btchs[b*2+1] = loc + btch_sz;
				if(DTV>0) printf(".Loc %d = %d, Loc %d = %d\n", b*2, trn_btchs[b*2], b*2+1, trn_btchs[b*2+1]); 
			} 
			trn_btchs[b*2]   = loc; 	
			trn_btchs[b*2+1] = Train_Sz; 
			if(DTV>0) printf("*Loc %d = %d, Loc %d = %d\n", b*2, trn_btchs[b*2], b*2+1, trn_btchs[b*2+1]);
		}
	
		else{  //Preference for homogenious to heterogenious, classes
			
			int b = 0, loc = 0;
			set_class_size(btch_sz); //Set the batchArray, to be the correct amount of pointers. 
			for (int i = 0; i < OPLEN; i++) { 
				int freq_len = Frequency[i];
				int uni_batches     = freq_len/btch_sz == 0? 1: freq_len/btch_sz;  //Amount of batches we are doing
				int batch_remainder = uni_batches == 1     ? 0: freq_len%btch_sz;  //Remainder to distrbute. 
				int split_remainder = batch_remainder==0   ? 0: batch_remainder/uni_batches; //Evenly split distrubution
				int tmp_btchSz = btch_sz+split_remainder; 												 //Update temporaliy.  
				if(DTV>0) printf("For class %d sized %d, btchSz = %d\n", i, freq_len, tmp_btchSz);
				for(int i=1; i < uni_batches; i++, b++, loc+=tmp_btchSz){
					trn_btchs[b*2]   = loc; 
					trn_btchs[b*2+1] = loc + tmp_btchSz; 
					if(DTV>0) printf(".Loc %d = %d, Loc %d = %d\n", b*2, trn_btchs[b*2], b*2+1, trn_btchs[b*2+1]);
				} 
				trn_btchs[b*2]   = loc; 	
				trn_btchs[b*2+1] = class_boundaries[i+1]; //Access all the way up to frequuency.
				if(DTV>0) printf("*Loc %d = %d, Loc %d = %d\n", b*2, trn_btchs[b*2], b*2+1, trn_btchs[b*2+1]);
				b++;
			}
 		}
	}

	int set_class_size(int btch_sz){
		total_btchs = 0;
		for (int i = 0; i < OPLEN; i++){ //Get all Sizes for all batches. BUT we are going to greedy fill the remainders.
			if((int)(Frequency[i] / btch_sz) < 1) total_btchs++;
			else total_btchs+= (int)(Frequency[i] / btch_sz); 
	
		} total_btchs*=2;
		trn_btchs = (int*)malloc((total_btchs)*sizeof(int));
		trn_btchs[0] = 0;
		
	}

/*=================================================================================================*/
/*Optimzied randomization!!*/
	void init_prng_threads() {
			#pragma omp parallel num_threads(OPLEN)
			{
				int tid = omp_get_thread_num();
				uint64_t seed = (uint64_t)time(NULL) + tid * 123456;
				s0 = seed | 1;
				s1 = (seed << 32) | 1;
			}
	}

	void shuffle_mb_postions(){
		for(int b = 0; b < total_btchs; b++){
			int j = xoroshiro128_next_range(0, total_btchs);
			j = j+2%2==0? j: j-1; //make sure it aligns up with a starting point. 
			 // Swap the pair of indices: trn_btchs[b] <-> trn_btchs[j] and trn_btchs[b+1] <-> trn_btchs[j+1]
			int tmp0 = trn_btchs[b], tmp1 = trn_btchs[b + 1];
			trn_btchs[b]     = trn_btchs[j]; trn_btchs[b + 1] = trn_btchs[j + 1];
			trn_btchs[j]     = tmp0;         trn_btchs[j + 1] = tmp1;
			
		}
	}

	void shuffle_PT_in_bounds(){
		#pragma omp parallel for num_threads(OPLEN)
		for (int c = 0; c < OPLEN; c++) {
			int start = class_boundaries[c];  	//LATER WE CAN REPLACE WITH GENERIC BOUNDS, SO NON-UNICLASS GETS EFFECTIVLEY SORTED!
			int end   = class_boundaries[c+1]; 
			for (int i = end - 1; i > start; i--) {
					int j = xoroshiro128_next_range(start, i);
					Numeric temp = Row_Access[i];
					Row_Access[i] = Row_Access[j];
					Row_Access[j] = temp;
			}
		}
	}

	void shuffle_PT(){
		for (int i = ROWS-1; i != 0; i--) {
				int j = xoroshiro128_next_range(0, i);
				Numeric temp = Row_Access[i];
				Row_Access[i] = Row_Access[j];
				Row_Access[j] = temp;
		}
	}

	inline int xoroshiro128_next_range(int min, int max) {
			return min + (int)(xoroshiro128_next() % (uint64_t)(max - min + 1));
	}
	
	inline uint64_t xoroshiro128_next() {
			uint64_t result = s0 + s1;
			uint64_t t = s1 ^ s0;
			s0 = ((s0 << 55) | (s0 >> (64 - 55))) ^ t ^ (t << 14);
			s1 = (t << 36) | (t >> (64 - 36));
			return result;
	}

/*=================================================================================================*/
/*MULTI-STYLE Ideas*/
  //0. Add additional for-loop that goes through, start and end, based off amount of classess per batch. Will be compiled out if =1. Or just un-rolled?
	//1.A Uniform
		 //1. Get amount of combinations, and minium amount of times a item shows up per set == minuim class size in a batch. 
		 //2. Adjust btch_sizing to get more even splits. Produce trn_btchs at those sizes: Either greed-fill, OR cut off un-even values: 
		 //3  Then, make array representaion off all combonations. Then search list to recombine, and make new sets. 
	//1.B Randomize sets: 
		 // Produce trn_btchs: Either greedy-fill, OR cut off un-even values: Might benfit, from adjust batchSize, we see more varience in splits. 
		 //4.2 Random: Just at random, iteraively move foward and combine sets at specific iterval. 	
	//2. Then, on randomize, you can still randomize Access_layer! As it will just randomize those indices. 
	//7. Theotricallyy we could come up with a intergrated way, to bypass prroducing mini-batches first. but for now it can remain for later. 


	//First create BatchStyle in Uni-class. Then, 
	int nCr(int n, int k) {
			// Compute C(n, k) efficiently using multiplicative method
			if (k > n) return 0;
			if (k > n - k) k = n - k;  // symmetry
			int res = 1;
			for (int i = 1; i <= k; i++) {
					res = res * (n - k + i) / i;
			}
			return res;
	} //nCr(n - 1, k - 1);  // how many combinations each element appears in

/*=================================================================================================*/


	void timer(FunctionPointer func, const char *msg){
		FILETIME start, end;
    ULARGE_INTEGER start_time, end_time;
    double cpu_time_used;

   
    GetSystemTimePreciseAsFileTime(&start);  /* Start timing */
		func();
    GetSystemTimePreciseAsFileTime(&end);

    /* Convert FILETIME to nanoseconds and compute difference */
    start_time.LowPart = start.dwLowDateTime;
    start_time.HighPart = start.dwHighDateTime;
    end_time.LowPart = end.dwLowDateTime;
    end_time.HighPart = end.dwHighDateTime;
    cpu_time_used = (double)(end_time.QuadPart - start_time.QuadPart) / 10000000.0; /* Convert 100-ns ticks to seconds */
    printf("      * %s :execution time: %.6f seconds\n", msg, cpu_time_used);
		
}
