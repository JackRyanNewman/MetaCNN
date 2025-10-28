#==================================================================================================
#Simulation Of C code, to see gains?
import numpy as np
from utils import *

def simulation(Cnfg, PV):
	msg = "   > Python Read,minMax, batches, sort 100*100"
	from multiprocessing import Pool
	with Pool(processes=1) as pool:
			status_code = pool.apply(timer, (simulate_c_file, msg, Cnfg, PV, True))
			if status_code != 0:
					pool.close()
					raise Exception(f"Worker failed with error code: {status_code}")

	# print("REMINDER RUN THESE TWO SEPERATELY AS THEY WILL BENFIT FROM CACHING!")
	# with Pool(processes=1) as pool:
	# 		status_code = pool.apply(timer, (simulate_c_file, msg, Cnfg, PV, False))
	# 		if status_code != 0:
	# 				pool.close()
	# 				raise Exception(f"Worker failed with error code: {status_code}")
	
def simulate_c_file(Cnfg,PV, tm_idv):
	tn_sz = int(Cnfg['ROWS']*Cnfg['TRNSZ'])
	dtype = np.double if Cnfg["DTP"] == 1 else np.float32
	if tm_idv: 

		data = load_and_reshape_data(Cnfg,dtype, PV) #Test effiency of load.
		miniMax(Cnfg, data, dtype, tn_sz)
		create_MB(Cnfg, data, dtype, tn_sz)
		Acess_IO = create_Access_IO(Cnfg, data, dtype, tn_sz)
		for i in range(0,100*100): np.random.shuffle(Acess_IO)
	else:
		data = timer(load_and_reshape_data, "        * load_and_reshape_data finished in", Cnfg, dtype, PV-1)
		timer(miniMax, "        * miniMax finished in", Cnfg, data, dtype, tn_sz)
		timer(create_MB, "        * create_MB finished in", Cnfg, data, dtype, tn_sz)
		Acess_IO = timer(create_Access_IO, "        * create_Access_IO+Frequncyy finished in", Cnfg, data, dtype, tn_sz)
		timer( np.random.shuffle, "        * Time to shuffle 1", Acess_IO)
	return 0

def load_and_reshape_data(Cnfg, dtype, PV):
	#Simulates a start-up process, reads the binary file, and reshapes the data for PyTorch/Numpy usage.
	rows =  Cnfg["ROWS"]; cols =  Cnfg["COLS"]; name =  Cnfg['FNAME']
	file_path = name[1:len(name)-1] 

	if PV>0: 	
		print("    --- New Process Started: Loading Data ---") # 1. Get File Info and Expected Shap
		print(f"        * Loading file: {file_path}\n        * Data type: {dtype}\n        * Expected shape: ({rows}, {cols})")
	
	try:
		data_1d = np.fromfile(file_path, dtype=dtype)         # 2. Load the 1D array using numpy.fromfile()
		if PV>0: print(f"        * Raw 1D array loaded. Shape: {data_1d.shape}")
		data_2d = data_1d.reshape(rows, cols) # 3. Reshape the 1D array to 2D (ROWS, COLS)
		expected_size = rows * cols
		if data_1d.size != expected_size: raise ValueError(
				f"        * Data size mismatch. Expected {expected_size} elements ({rows}x{cols}), "
				f"        * but found {data_1d.size} in the file. Check ROWS/COLS or data type."
		)

	except ValueError as e: p_error(f"        * ERROR: Not good diminsions. {e}")
	except Exception as e: p_error(f"        * ERROR: Reshape failed. {e}")
	return data_2d


def miniMax(Cnfg, data, dtype, tn_sz):
	#In-place feature scaling for all columns except the last (classifier).
	#Scales features to [-1, 1] using min/max from the first train_fraction of rows.
	ROWS, COLS = data.shape
	features = data[:, :-1].astype(dtype, copy=False) # all feature columns except classifier
	train_features = features[:tn_sz]
	L = np.min(train_features, axis=0).astype(dtype)
	U = np.max(train_features, axis=0).astype(dtype)
	denom = U - L; denom[denom == 0] = 1.0  # avoid div by zero
	data[:, :-1] = -1.0 + 2.0 * (features - L) / denom # In-place scale: [-1, 1]
	
#JUST ALTER MY C CODE. 
def create_MB(Cnfg, data, dtype, tn_sz):
		btchSz = Cnfg["btchSz"]; batchStyle = Cnfg["batchStyle"] 
		ROWS, COLS = data.shape; OPLEN = ROWS-1
		btch_sz = int(tn_sz * btchSz) if btchSz < 1 else tn_sz if btchSz >= tn_sz else btchSz

	
		# Create minibatch start/stop indices
		if btch_sz == tn_sz or batchStyle == 0:
				arrSz = 2 if tn_sz == btch_sz else (tn_sz // btch_sz) * 2
				trn_btchs = np.zeros(arrSz, dtype=int)
				b = 0; loc = 0
				while b < (arrSz // 2) - 1:
						trn_btchs[b * 2] = loc
						trn_btchs[b * 2 + 1] = loc + btch_sz
						b += 1
						loc += btch_sz
				trn_btchs[b * 2] = loc
				trn_btchs[b * 2 + 1] = tn_sz
				
		else: trn_btchs = np.array([0, tn_sz], dtype=int)
		return trn_btchs


def create_Access_IO(Cnfg, data, dtype, tn_sz):
		ROWS, COLS = data.shape; OPLEN = ROWS-1
		labels = data[:, -1]; int_labels = labels.astype(int)

		classifier_freq = np.bincount(int_labels, minlength=OPLEN).astype(dtype) 	# Compute frequency of each classifier
		class_boundaries = np.cumsum([0] + classifier_freq.tolist()) 							# Compute class boundaries (cumulative sum of frequencies)
		x = 0
		for i in range(0, 20): x+=class_boundaries[i]

		return np.argsort(int_labels) 		#Arg sort computes the indices that would sort the array. Create ACCES_IO: indices sorted by classifier
		

