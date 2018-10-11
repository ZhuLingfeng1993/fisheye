#include "opencv2/opencv.hpp"
//#include "direct.h"
//#include "io.h"

#include <fstream>
#include <iostream>
using namespace std;
using namespace cv;

extern void distortPoints(InputArray undistorted, OutputArray distorted,
  InputArray K, InputArray D, InputArray R, InputArray P);
  
//void myFisheyeCalib(string calibImgFolder, string cameraParamsFileName) {};



extern void testDistortPoints();

