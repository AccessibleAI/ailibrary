## Tensorflow Lite Model Converter
### IoT models easily deployed from your .h5 weights file

Simply specify: 

1. Input Model 
2. Output Model 

## How It Works

Upon the start-up the application takes parameters and converts the input model to optimized tensorflow lite version for edge device usage

## Running

provide the "input" model name (without the .h5) and the name of the desired output model "model" 

example 
python3 convert.py \
       --model=my_lite_model\
       --input=current_model
       
will produce a file /cnvrg/my_lite_model.tflite from /cnvrg/current_model.h5 


```
![alt text](https://github.com/vvagias/cnvrg_ai_library_extras/blob/main/tensorflow-lite/tf_lite_ailib.png?raw=true)
![alt text](https://github.com/vvagias/cnvrg_ai_library_extras/blob/main/tensorflow-lite/tf_lite.png?raw=true)


