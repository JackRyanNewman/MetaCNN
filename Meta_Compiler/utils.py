
import sys, re, os, shutil, traceback
import time, sys, traceback

def timer(func, msg, *args, **kwargs):
    """
    Time the execution of a function
    
    Args:
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        tuple: (result, execution_time)
    """
    start_time = time.perf_counter()  # High precision timer
    result = func(*args, **kwargs)
    
    print(f"{msg} {time.perf_counter()-start_time:.6f} seconds")
    return result



#==================================================================================================
#Helpers
def p_error(mssg):
    tb = sys.exc_info()[2]
    print(f"   Error: {mssg}")
    print("   Stack trace:")
    traceback.print_tb(tb)
    sys.exit(1)