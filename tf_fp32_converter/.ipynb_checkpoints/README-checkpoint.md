
## Tensorflow h5 --> FP32 Model Converter
### FP32 optimized model from your .h5 weights file
 
![alt text](https://github.com/vvagias/cnvrg_ai_library_extras/blob/main/tf_fp32_converter/tf_fp32.png?raw=true)
 


## How It Works

Upon the start-up the application takes parameters and converts the input model to optimized tensorflow lite version for edge device usage

## Running

 
Simply specify: 

1. Input Model 
2. Output Model 
 
 example if you have "model.h5" and would like to get "converted.pb" then you would provide input = model , model = converted
 you will have your files in the /cnvrg/ directory ex.) /cnvrg/converted.pb 
  
