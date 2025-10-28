#Jack Newman.
#NOTE: This file was meant to scrum, the time for conventiton, 
#optimizatiion, and readbility is really in low comparsion to my c files. 

#==================================================================================================
import numpy as np
from utils import *
from configs import *
from bench_marking import *
import inspect

 

PV = 1 #Python verbosity+testing. 
#==================================================================================================
def main():
	try: 	
		Cnfg = create_Cnfg(r"01_Dataset/mnist_train")          #Currently statically made, based off mnsit. 
		pre_compile_c(Cnfg) 
		create_bin(Cnfg)
		if PV>0: simulation(Cnfg,PV)
	except Exception as e: delete_file_on_failure(e)
	return 0
#==================================================================================================
#Pre-process c file. 

def pre_compile_c(Cnfg): 
	#Future Feature 1: Find OPLEN, ROWS, VALLEN. Pass kernel dimsions. 
	#Future Feature 2: Create Filter data_set your own way, and make compy. 
	shutil.copy2(r"CNN.h", r"CNN_c.h") # Copy original CNN.H to CNN_c.H
	with open(r"CNN_c.h", "r") as f: content = f.read() # Read the file

	pattern = r"/\*@D.*?//@E"; 
	replace_str = "\n".join([f"#define {k} {v}" for k, v in  Cnfg.items()]); 
	new_content = re.sub(pattern, f"/*@S*/\n{replace_str}\n  //@E", content, flags=re.DOTALL)
	with open(r"CNN_c.h", "w") as f: f.write(new_content) # Write it back

def create_bin( Cnfg):

	num = Cnfg["DTP"]; name = Cnfg['FNAME']; 
	dtype = np.double if num == 1 else np.float32
# Remove the number from input file, keep it for output file
	input_file = name[1:len(name)-6]+ ".csv"  # Remove the number
	binName = name[1:len(name)-1]

	if os.path.exists(binName): os.remove(binName) #Delete just incase. 
	(np.loadtxt(input_file, delimiter=",", dtype=dtype)).tofile(binName)

def delete_file_on_failure(e):
	print(f"ERROR:{e}")
	flDel = "CNN_c.h"
	if os.path.exists(flDel):
		os.remove(flDel)
		print(f"   File '{flDel}' deleted.{e}")
	else: print(f"   File '{flDel}' does not exist.{e}")
	p_error(f"Problem Reading Config {e}")





if __name__ == "__main__": sys.exit(main())
