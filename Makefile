CFLAGS = -O3 -march=native -mtune=native -mavx2 -mfma \
         -ffast-math -fno-trapping-math -fno-math-errno \
         -fstrict-aliasing -fopenmp \
         -D PROCESS_GUARD=1 -I.
#-fno-trapping-math -fno-math-errno 
LDFLAGS = -fopenmp -static
CC=gcc 

PROGS=CNN
ENVR = OMP_STACKSIZE=128k
DYNAMIC_FILE=CNN_c.h
FILES_TO_CP=CNN.c CNN_c.h
PYTHON_SCRIPT= Meta_Compiler\A_Pre_Proccesor.py


all:
	@echo *1: Running Python script to generate temporary files...
	@python $(PYTHON_SCRIPT)
	@if %ERRORLEVEL% NEQ 0 exit /b 1

	@echo *2: Compiling binaries...
	@$(CC) -o CNN.exe $(FILES_TO_CP) $(CFLAGS)

	@echo *3: Cleaning off temporary header files...
	@rem Windows: delete if needed: del /Q $(DYNAMIC_FILE)

	@echo *4: Running the CNN program...
	@ CNN.exe

pre_comp:
	@echo "Step 1: Running Python script to generate temporary files..."
	python $(PYTHON_SCRIPT) || echo "Python script failed!"; exit 1; 

bin_comp: 
	@echo "Step 2: Compiling binaries..."
	$(CC) -o CNN $(FILES_TO_CP) $(CFLAGS) $(LDFLAGS)

 
run: 
	@echo * 4: Running the CNN program...
	./CNN




# Clean only binaries (now part of run-cnn)
clean:
	@echo "Cleaning previous binaries..."
	$(RM) $(PROGS) *.o
	$(RM) -rf *.dSYM

.PHONY: all run-cnn clean

# -O3: Enables aggressive optimization for maximum performance, prioritizing speed over code size.
# -march=native: Optimizes for the host CPU architecture, enabling instructions specific to the machine compiling the code.
# -mtune=native: Tunes performance for the host CPU, optimizing instruction scheduling and other low-level details.
# -mavx2: Enables Advanced Vector Extensions 2 (AVX2) instructions for SIMD (Single Instruction, Multiple Data) operations, improving performance on compatible CPUs.
# -mfma: Enables Fused Multiply-Add instructions, which combine multiplication and addition for better floating-point performance.
# -ffast-math: Allows the compiler to make assumptions about floating-point math (e.g., ignoring IEEE compliance) to optimize performance, potentially at the cost of precision.
# -fno-trapping-math: Disables floating-point exception trapping, assuming no floating-point exceptions occur, for faster execution.
# -fno-math-errno: Prevents setting errno for math functions, reducing overhead in error checking.
# -fstrict-aliasing: Enables strict type-based aliasing optimizations, assuming pointers of different types don’t alias, which can improve performance but requires careful coding to avoid undefined behavior.
# - fscrict alising is on by default of 02
# -funroll-loops: Unrolls loops to reduce loop overhead, potentially increasing code size but improving performance for tight loops.
# -fopenmp: Enables OpenMP support for parallel processing, allowing multi-threaded execution on multi-core systems.
# -D PROCESS_GUARD=1: Defines a preprocessor macro PROCESS_GUARD with value 1, likely used for conditional compilation (e.g., enabling a specific feature or guard in the code).
# -I.: Adds the current directory (.) to the include path for header files.

# LDFLAGS (Linker Flags)
# -fopenmp: Links the OpenMP runtime library to support parallel execution.
# -static: Links libraries statically, creating a standalone executable that doesn’t depend on shared libraries at runtime.