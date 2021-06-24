#include "hogDefines.h"


int main(int argc, const char* argv[])
{
	string pathData;
	String keys = "{@arg1||}";
	CommandLineParser cmd(argc, argv, keys);
	//cout << "Enter Tes' Data directory path" << endl;
	//cin >> pathData;
//	int test_pic_count = 17;
	
	pathData = cmd.get<String>(0);
	
	int testCount = 2*TEST_NUM;
	
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
	
	int i = 0;
	int test_gold;
	std::vector<float> descriptors;
	int counter = 0;
	
	for (int i = 1; i < testCount; ++i)
	{
		stringstream filePath;
		filePath << pathData << "/" << i << ".jpg";
		Mat testImg = imread(filePath.str(), 1);
		if(i < TEST_NUM)
			test_gold = -1;
		else
			test_gold = 1;
		hogTest.compute(testImg, descriptors, Size(0, 0), Size(0, 0));
		//cout<<"compute"<<endl;
		float a = svmLoad->predict(descriptors);
		descriptors.clear();
		if(a == test_gold)
			counter++;
	
	}
	cout<<counter<<endl;
	cout<<"Accuracy: "<<((float)counter/(float)(2*TEST_NUM))*100<<"%"<<endl;
}

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
