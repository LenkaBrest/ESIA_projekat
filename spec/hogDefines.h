#ifndef HOGDEFINES_H
#define HOGDEFINES_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/ml.hpp"
#include <ctype.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
//#include "precomp.hpp"
//#include "cascadedetect.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/hal/intrin.hpp"
//#include "opencl_kernels_objdetect.hpp"
#include <fstream>
#include <ctime>

#define IMG_NUM 5417 /// number of images in set
#define TEST_NUM 542
#define HOW_LONG 0 /// wait until key or X is pressed, other value is in milisecs

using namespace std;
using namespace cv;
using namespace cv::ml;


void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void detectFaces(Mat &img, HOGDescriptor &hogTest, Ptr<SVM> svmLoad);

#endif
