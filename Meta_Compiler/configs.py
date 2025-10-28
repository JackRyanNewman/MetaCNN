def create_Cnfg(path, readFile=60000): #Static Config
  c = {
   # 1. Data Configuration: How the file is set up.
    "DTP":     1,                          # -F 1=double, 0=flaoat
    "X":       28,                         # -F Input image width  
    "Y":       28,                         # -F Input image height
    "OPLEN":   10,                         # -F Num outputs the network will have
    "IOLEN":   "X*Y",                      # -G Num inputs flattened + 1 spot for type of output
    "ROWS":    60000,                      # -F DataSet Size
    "COLS":    "IOLEN+1",                  # -G Amount of cols per row
    "FNAME":   "path",                       # -F (DTP inserted directly)

  # 2. Computational Configurations
    "Rand":    0,                          # -F Randomize data-set trainning 
    "TRV":     1,                          # -F Verbsoity during training 0 = silent.
    "DTV":     0,                          # -F Verboisty for dt informmation.
    "TRNSZ":   0.8,                        # -F size of training data from given set
    "SIMDID":  256,                        # -F
    "IO_THRDS": "Rows*THR",                # -F Amount of threads
    "CP_THRDS": 1,                         # -F
    "GPUON":   0,                          # -F

  # 3. Training Info
    "initWE":     0.1,                     # -w Weight initialization parameter
    "alpha":      0.01,                    # -a Learning rate for gradient descent
    "lambda":     0,                       # -l Regularization parameter
    "btchSz":     .012,                     # -m Batch size (0 for full batch)
    "epochLimit": 1000,                    # -e Epoch limit for gradient descent
    "batchStyle": 1,                       # Batch style (1=uni-class)
    "even_dis":   0,                       # if 1=uni-class, 2- multi classes. 
  
  # 4. TRAINING CONFIGURATION:
      #   LOSS: str              # "categorical_crossentropy", "sparse_categorical_crossentropy", "mse"
      #   METRICS: list          # ["accuracy", "precision", "recall"]
      #   WEIGHT_DECAY: float    # L2 regularization factor
      #   DROPOUT_RATE: float    # Global dropout rate
  }
  # Second pass - calculated values
  c['FNAME'] = f'"{path}{c["DTP"]}.bin"'
  c["IOLEN"] = c['X'] * c['Y']
  c["COLS"] = c['IOLEN'] + 1
  c["IO_THRDS"] = 16

  #Conv{ type: "conv", filters: int, kernel_size: int|tuple, stride: int|tuple, padding: int|tuple|"same"|"valid", activation: str, use_bias: bool }
  #BatchNorm{ type: "batchnorm", axis: int }
  #Pool{ type: "pool", pool_type: str, kernel_size: int|tuple, stride: int|tuple, padding: int|tuple }
  #Flatten{ type: "flatten", input_size: any }
  #Dense{ type: "dense", units: int, activation: str, use_bias: bool }
  #Dropout{ type: "dropout", rate: float }
  #Residual{ type: "residual", blocks: int, filters: list }
  return c


"""
CNN CONFIGURATION DICTIONARY SPECIFICATION

LAYER TYPES AND THEIR PARAMETERS:
---------------------------------
  CONVOLUTIONAL LAYER:
      type: "conv"
      filters: int                # Number of output filters/channels
      kernel_size: int or tuple   # Convolution window size
      stride: int or tuple        # Stride of convolution
      padding:                    # int or tuple or "same" or "valid"
      activation: str             # "relu", "sigmoid", "tanh", "leaky_relu", None
      use_bias: bool              # Whether to use bias term

  BATCH NORMALIZATION LAYER:
      type: "batchnorm"
      axis: int           # Axis to normalize    

  POOLING LAYER:
      type: "pool" 
      pool_type: str      # "max", "average", "global_max", "global_average"
      kernel_size: int or tuple  # Pooling window size
      stride: int or tuple      # Stride of pooling
      padding: int or tuple    # Padding for pooling

  FLATTEN LAYER:
      type: "flatten"
      input_size: any     # Usually None (calculated automatically)
      
  DENSE/FULLY CONNECTED LAYER:
      type: "dense"
      units: int          # Number of output units
      activation: str     # Activation function
      use_bias: bool      # Whether to use bias term

  DROPOUT LAYER:
      type: "dropout" 
      rate: float         # Dropout rate (0.0-1.0)

  RESIDUAL LAYER:
      type: "residual"
      blocks: int         # Number of residual blocks
      filters: list       # Filters for each block


"""














