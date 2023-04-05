# YOLOv5 Segmenation Implementation in C# and OnnxRuntime

## Notice
This repository is for only yolov5-seg inference using onnx **(NOT FOR TRAIN)**

## How to use
1. You have to train in python or libtorch

2. Convert pt or pth file to onnx

3. Load onnx file and insert images

4. **YOU CAN USE IT!!**


## Changes
OpenCvSharp3 => OpenCvSharp4 4.2.0.20191223

Microsoft.ML.OnnxRuntime 1.7.0 => Microsoft.ML.OnnxRuntime.GPU 1.11.0

Microsoft.ML.OnnxRuntime.Managed is upgraded to 1.11.0 automatically

OpenCvSharpExtern.dll and OpenCvSharpExtern.pdb is necessary for build

You can download these files on https://github.com/shimat/opencvsharp/releases with your vesrion

## Original code
https://github.com/singetta/OnnxSample

**Permission to share granted by the original author**
