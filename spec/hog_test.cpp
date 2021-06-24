#include "hogDefines.h"


int main(int argc, const char* argv[])
{
	String pathData;

	String keys = "{@arg1||}";
	CommandLineParser cmd(argc, argv, keys);
	//cout << "Enter Tes' Data directory path" << endl;
	//cin >> pathData;
//	int test_pic_count = 17;
	
	pathData = cmd.get<String>(0);
	
	Ptr<SVM> svmLoad;
	svmLoad = SVM::load("hogSVM.xml");
	//Mat loadSVMMat = svmLoad->getSupportVectors();
	vector<float> loadSVMvector;
	
	HOGDescriptor hogTest;
	hogTest.winSize = Size(64, 64);
	hogTest.blockSize = Size(4, 4);
	hogTest.blockStride = Size(2, 2);
	hogTest.cellSize = Size(2, 2);

	get_svm_detector(svmLoad, loadSVMvector);
	hogTest.setSVMDetector(loadSVMvector);

	cout<<loadSVMvector.size()<<endl;
	cout<<"load svm"<<endl;
	
	int testCount = TEST_NUM;

	vector<int> test;
	
	//for (int i = 0; i < test_lst.size(); ++i)
	//{
   //for(int j=0; j<10; j++)
   //{
	//for (int i = 1; i < testCount; ++i)
	//{
		stringstream filePath;
	//	filePath << pathData << "/" << i << ".jpg";
		filePath << pathData << "/" << "7.jpg";
		//filePathName << pathData;
		cout<<filePath.str()<<endl;
		Mat testImg = imread(filePath.str(), 1);
		
		//cout<<"img read"<<endl;
		
		detectFaces(testImg, hogTest, svmLoad);
		
		imshow( "Test image", testImg );
		waitKey(0);
	
		
	//}
   //}
	
}

// Following subroutine "get_svm_detector" to convert SVM parameters to vector floats value has been taken from the mentioned website
// https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;

	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);

	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||

		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));

	CV_Assert(sv.type() == CV_32F);

	hog_detector.clear();
	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

void detectFaces(Mat &img, HOGDescriptor &hogTest, Ptr<SVM> svmLoad)
{
	
	
	int i = 1;
	float a;
	Mat part;
	std::vector<float> descriptors;
	do
	{
		hogTest.winSize = Size(64*i, 64*i);
		hogTest.blockSize = Size(4*i, 4*i);
		hogTest.blockStride = Size(2*i, 2*i);
		hogTest.cellSize = Size(2*i, 2*i);
		cout<<"set parameters"<<endl;
	
		for(int j=0; j+64*i<img.rows; j+=32*i)
		{
			cout<<"first for: "<<j<<endl;
			for(int k=0; k+64*i<img.cols; k+=32*i)
			{
				cout<<"second for: "<<k<<endl;
				part = img(Range(j, j+64*i), Range(k, k+64*i));
				cout<<part.rows<<endl;
				cout<<part.cols<<endl;
				/*HOGCache cache(this, part, Size(0, 0), Size(0, 0), nwindows == 0, cacheStride);
				const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();
    descriptors.resize(dsize*nwindows);

    // for each window
    for( size_t i = 0; i < nwindows; i++ )
    {
        float* descriptor = &descriptors[i*dsize];

        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
//            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }

        for( int j = 0; j < nblocks; j++ )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;

            float* dst = descriptor + bj.histOfs;
            const float* src = cache.getBlock(pt, dst);
            if( src != dst )
                memcpy(dst, src, blockHistogramSize * sizeof(float));*/
				hogTest.compute(part, descriptors, Size(0, 0), Size(0, 0));
				cout<<"deskriptor size: "<<descriptors.size()<<endl;
				a = svmLoad->predict(descriptors);
				cout<<"detect"<<endl;
				if(a == -1)
					rectangle(img, Point(k, j), Point(k+64*i, j+64*i), Scalar(0, 255, 0));
				cout<<"draw rectangle"<<endl;
				descriptors.clear();
					
					
			}
		}
		cout<<"i = "<<i<<endl;
		i++;
	}while(64*i < min(img.cols, img.rows));
}

