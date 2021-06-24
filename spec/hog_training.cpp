#include "hogDefines.h"


vector< Mat > img_pos_lst, img_neg_list, test_lst;


int main(int argc, const char* argv[])
{
	String pathData;
	int posCount, negCount, testCount;
	//cout << "Enter Positive Data directory path" << endl;
	//cin >> pathData;

	String keys = "{@arg1||}";
	CommandLineParser cmd(argc, argv, keys);
	pathData = cmd.get<String>(0);

	srand(time(NULL));

	posCount = IMG_NUM;
	negCount = IMG_NUM;
	
	testCount = 0;

	for(int i=1; i < posCount; ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "/"<< "pveimages" << "/" << i << ".jpg";
		Mat img = imread(filePathName.str(),1);
		if (img.empty())
		{ 
			return -1;
		}

		
		if(testCount < TEST_NUM && rand()%2 == 1)
		{
			stringstream testPathName;
			stringstream accuracyPathName;
			testPathName << pathData << "/" << "test" << "/" << testCount+1 << ".jpg";
			imwrite(testPathName.str(), img);
			
			resize(img, img, Size(64, 64));
			accuracyPathName << pathData << "/" << "accuracy" << "/" << testCount+1 << ".jpg";
			imwrite(accuracyPathName.str(), img);
			++testCount;
		}
		else
		{
			resize(img, img, Size(64, 64));
			img_pos_lst.push_back(img.clone());
		}

	}
	
	cout<<"Get out"<<endl;

	for (int i = 1; i < negCount; ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "/" << "nveimages" << "/" << "TrainNeg" <<"/"<<i<< ".jpg";
		Mat img = imread(filePathName.str(), 1);
		if (img.empty())
		{
			return -1;
		}
		

		
		if(testCount < TEST_NUM*2 && rand()%2 == 1)
		{
			stringstream testPathName;
			stringstream accuracyPathName;
			testPathName << pathData << "/" << "test" << "/" << testCount+1 << ".jpg";
			imwrite(testPathName.str(), img);
			
			resize(img, img, Size(64, 64));
			accuracyPathName << pathData << "/" << "accuracy" << "/" << testCount+1 << ".jpg";
			imwrite(accuracyPathName.str(), img);
			++testCount;
		}
		else
		{
			resize(img, img, Size(64, 64));
			img_neg_list.push_back(img.clone());
		}
	}

	cout<<"Get out of TrainNeg"<<endl;
	


	HOGDescriptor hog;
	Mat gradMat;
	vector<int> labelsMat;
	Mat trainingDataMat;
	std::vector<float> descriptors;

	hog.winSize = Size(64, 64);
	hog.blockSize = Size(4, 4);
	hog.blockStride = Size(2, 2);
	hog.cellSize = Size(2, 2);

	
	for (int i = 0; i < img_neg_list.size(); ++i)
	{
		hog.compute(img_neg_list[i], descriptors, Size(0, 0), Size(0, 0));
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);
		trainingDataMat.push_back(descMat);
		descriptors.clear();
		labelsMat.push_back(1);
	}
	
	
	cout<<"neg_lst"<<endl;
	
	//For positive Data
	cout<<"before making pos feature"<<endl;
	for (int i = 0; i < img_pos_lst.size(); ++i)
	{
		hog.compute(img_pos_lst[i], descriptors, Size(0, 0), Size(0, 0));	
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);		
		trainingDataMat.push_back(descMat);
		descriptors.clear();

		labelsMat.push_back(-1);
		//cout<<i<<endl;
	}
	cout<<"after making pos feature"<<endl;


	Ptr<SVM> svm = SVM::create();
	
	//svm->setCoef0( 0.0 );
    	 
        
        //svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
        svm->setType(SVM::C_SVC);
        svm->setKernel( SVM::LINEAR );
        //svm->setGamma( 0 ); //3
        svm->setDegree( 1 );//3
    	svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
    	//svm->setNu( 0.5 );
    	//svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    	svm->setC( 0.01 ); // From paper, soft classifier
	//svm->setType(SVM::C_SVC);
	cout<<"set svm coef"<<endl;
	cout<<trainingDataMat.size()<<endl;
	cout<<labelsMat.size()<<endl;
	Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	
	svm->train(td);
	cout<<"trained"<<endl;
	
	svm->save("hogSVM.xml");
	
return 0;
}
