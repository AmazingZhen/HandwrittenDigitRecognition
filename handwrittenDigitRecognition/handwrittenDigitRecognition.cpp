// handwrittenDigitRecognition.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <time.h>

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

int random(int min, int max);
int reverseDigit(int i);
vector<Mat> readDigits(string filepath);
Mat_<int> readLabels(string filePath);
Mat_<float> extractFeatures(const vector<Mat> &trainDigits);
Ptr<SVM> trainSVM(const Mat_<float> &dataMat, const Mat_<int> &labelMat);
Ptr<Boost> trainBoost(const Mat_<float> &dataMat, const Mat_<int> &labelMat);
Mat_<int> getPredictLabels(Ptr<SVM> svm, const Mat_<float> &testMat);
Mat_<int> getPredictLabels(Ptr<Boost> boost, const Mat_<float> &testMat);
float getAccuracy(const Mat_<int> &testLabels, const Mat_<int> &predictLabels);

int main(int argc, char* argv[]) {
	ifstream file1, file2;

	vector<Mat> testDigits = readDigits("dataset/t10k-images.idx3-ubyte");
	Mat_<int> testLabels = readLabels("dataset/t10k-labels.idx1-ubyte");
	Mat_<float> testFeatures = extractFeatures(testDigits);

	file1.open("dataset/mysvm.xml");
	file2.open("dataset/myboost.xml");

	if (!file1.is_open() || !file2.is_open()) {
		vector<Mat> trainDigits = readDigits("dataset/train-images.idx3-ubyte");
		Mat_<int> trainLabels = readLabels("dataset/train-labels.idx1-ubyte");
		Mat_<float> trainFeatures = extractFeatures(trainDigits);
		trainDigits.clear();

		if (!file1.is_open()) {
			Ptr<SVM> svm = trainSVM(trainFeatures, trainLabels);
			svm->save("dataset/mysvm.xml");
			trainFeatures.release();
			trainLabels.release();

			Mat_<int> predictLabels = getPredictLabels(svm, testFeatures);

			float result = getAccuracy(testLabels, predictLabels);
			cout << "The accuracy of the SVM is " << result << ", totally " << predictLabels.size().height << " images." << endl;


			cout << "\nWould you want to see some SVM test cases?" << endl;
			cout << "Press y to see, n to skip." << endl;
			char t;
			while (cin >> t) {
				if (t == 'y') {
					cout << "Totally 20 images." << endl;
					srand(time(0));

					for (int i = 0; i < 20; i++) {
						int chosen_index = random(0, predictLabels.size().height);
						imshow("Test", testDigits[chosen_index]);
						cout << "The predict result is " << predictLabels.at<int>(chosen_index, 0) << endl;
						cout << "Press any key to see next image." << endl;
						waitKey(0);
					}
					break;
				}
				if (t == 'n') {
					break;
				}

				cout << "Wrong input!" << endl;
			}
		}

		if (!file2.is_open()) {
			Ptr<Boost> boost = trainBoost(trainFeatures, trainLabels);
			boost->save("dataset/myboost.xml");
			trainFeatures.release();
			trainLabels.release();

			Mat_<int> predictLabels = getPredictLabels(boost, testFeatures);

			float result = getAccuracy(testLabels, predictLabels);
			cout << "The accuracy of the Adaboost is " << result << ", totally " << predictLabels.size().height<< " images." << endl;


			cout << "\nWould you want to see some Adaboost test cases?" << endl;
			cout << "Press y to see, n to skip." << endl;
			char t;
			while (cin >> t) {
				if (t == 'y') {
					cout << "Totally 20 images." << endl;
					srand(time(0));

					for (int i = 0; i < 20; i++) {
						int chosen_index = random(0, predictLabels.size().height);
						imshow("Test", testDigits[chosen_index]);
						cout << "The predict result is " << predictLabels.at<int>(chosen_index, 0) << endl;
						cout << "Press any key to see next image." << endl;
						waitKey(0);
					}
					break;
				}
				if (t == 'n') {
					break;
				}

				cout << "Wrong input!" << endl;
			}
		}
	}
	else {
		if (file1.is_open()) {
			Ptr<SVM> svm = Algorithm::load<SVM>("dataset/mysvm.xml");

			Mat_<int> predictLabels = getPredictLabels(svm, testFeatures);

			float result = getAccuracy(testLabels, predictLabels);
			cout << "The accuracy of the SVM is " << result << ", totally " << predictLabels.size().height << " images." << endl;



			cout << "\nWould you want to see some SVM test cases?" << endl;
			cout << "Press y to see, n to skip." << endl;
			char t;
			while (cin >> t) {
				if (t == 'y') {
					cout << "Totally 20 images." << endl;
					srand(time(0));

					for (int i = 0; i < 20; i++) {
						int chosen_index = random(0, predictLabels.size().height);
						imshow("Test", testDigits[chosen_index]);
						cout << "The predict result is " << predictLabels.at<int>(chosen_index, 0) << endl;
						cout << "Press any key to see next image." << endl;
						waitKey(0);
					}
					break;
				}
				if (t == 'n') {
					break;
				}

				cout << "Wrong input!" << endl;
			}
		}

		if (file2.is_open()) {
			Ptr<Boost> boost = Algorithm::load<Boost>("dataset/myboost.xml");

			Mat_<int> predictLabels = getPredictLabels(boost, testFeatures);

			float result = getAccuracy(testLabels, predictLabels);
			cout << "The accuracy of the Adaboost is " << result << ", totally " << predictLabels.size().height << " images." << endl;



			cout << "\nWould you want to see some Adaboost test cases?" << endl;
			cout << "Press y to see, n to skip." << endl;
			char t;
			while (cin >> t) {
				if (t == 'y') {
					cout << "Totally 20 images." << endl;
					srand(time(0));

					for (int i = 0; i < 20; i++) {
						int chosen_index = random(0, predictLabels.size().height);
						imshow("Test", testDigits[chosen_index]);
						cout << "The predict result is " << predictLabels.at<int>(chosen_index, 0) << endl;
						cout << "Press any key to see next image." << endl;
						waitKey(0);
					}
					break;
				}
				if (t == 'n') {
					break;
				}

				cout << "Wrong input!" << endl;
			}
		}

	}

	file1.close();
	file2.close();

	//file.open("dataset/myboost.xml");

	//if (!file.is_open()) {
	//	vector<Mat> trainDigits = readDigits("dataset/train-images.idx3-ubyte");
	//	vector<Mat> testDigits = readDigits("dataset/t10k-images.idx3-ubyte");
	//	Mat_<int> trainLabels = readLabels("dataset/train-labels.idx1-ubyte");
	//	Mat_<int> testLabels = readLabels("dataset/t10k-labels.idx1-ubyte");

	//	Mat_<float> trainFeatures = extractFeatures(trainDigits);
	//	Mat_<float> testFeatures = extractFeatures(testDigits);
	//	trainDigits.clear();
	//	testDigits.clear();

	//	Ptr<Boost> boost = trainBoost(trainFeatures, trainLabels);
	//	boost->save("dataset/myboost.xml");

	//	//trainFeatures.release();
	//	//trainLabels.release();

	//	//Mat_<int> predictLabels = getPredictLabels(svm, testFeatures);
	//	//testFeatures.release();

	//	//float result = getAccuracy(testLabels, predictLabels);
	//	//cout << "The accuracy of the SVM is " << result << endl;
	//}
	//else {
	//	vector<Mat> testDigits = readDigits("dataset/t10k-images.idx3-ubyte");
	//	Mat_<float> testFeatures = extractFeatures(testDigits);

	//	testDigits.clear();

	//	Ptr<Boost> boost = Algorithm::load<Boost>("dataset/myboost.xml");

	//	Mat_<int> predictLabels = getPredictLabels(boost, testFeatures);
	//	testFeatures.release();

	//	Mat_<int> testLabels = readLabels("dataset/t10k-labels.idx1-ubyte");

	//	float result = getAccuracy(testLabels, predictLabels);
	//	cout << "The accuracy of the adaboost is " << result << endl;
	//}

	cout << "\nPress any key to quit" << endl;
	char t;
	cin >> t;

	return 0;
}

Ptr<SVM> trainSVM(const Mat_<float> &dataMat, const Mat_<int> &labelMat) {
	assert(dataMat.rows == labelMat.rows);
	assert(labelMat.cols == 1);

	cout << "SVM training start." << endl;

	Ptr<SVM> svm = SVM::create();

	svm->setType(SVM::Types::C_SVC);
	svm->setKernel(SVM::KernelTypes::LINEAR);
	//svm->setDegree(0);  // for poly
	//svm->setGamma(20);  // for poly/rbf/sigmoid
	//svm->setCoef0(0);   // for poly/sigmoid
	//svm->setC(1);       // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	//svm->setNu(0);      // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
	//svm->setP(0);       // for CV_SVM_EPS_SVR

	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, FLT_EPSILON));

	Ptr<TrainData> trainData = TrainData::create(dataMat, SampleTypes::ROW_SAMPLE, labelMat);
	svm->trainAuto(trainData);

	cout << "SVM Training Completed." << endl;

	return svm;
}

Ptr<Boost> trainBoost(const Mat_<float> &dataMat, const Mat_<int> &labelMat) {
	int ntrain_samples = dataMat.rows;
	int var_count = dataMat.cols;
	int class_count = 10;

	// Unroll the database type mask.
	// Based on https://github.com/Itseez/opencv/blob/master/samples/cpp/letter_recog.cpp
	// If the label of current digit is 5
	// The new data will be : 
	// {{feature_vector}, 0}
	// {{feature_vector}, 1},
	// ... , 
	// {{feature_vector}, 9}
	// The new response will be : 0,0,0,0,0,1,0,0,0,0

	Mat new_data(ntrain_samples*class_count, var_count + 1, CV_32F);
	Mat new_responses(ntrain_samples*class_count, 1, CV_32S);

	cout << "Unrolling the database..." << endl;
	for (int i = 0; i < ntrain_samples; i++) {
		const float* data_row = dataMat.ptr<float>(i);
		for (int j = 0; j < class_count; j++) {
			float* new_data_row = (float*)new_data.ptr<float>(i * class_count + j);
			memcpy(new_data_row, data_row, var_count * sizeof(data_row[0]));
			new_data_row[var_count] = (float)j;
			new_responses.at<int>(i*class_count + j) = (labelMat.at<int>(i) == j);
		}
	}

	Mat var_type(1, var_count + 2, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count + 1) = VAR_CATEGORICAL;

	Ptr<TrainData> tdata = TrainData::create(new_data, ROW_SAMPLE, new_responses,
		noArray(), noArray(), noArray(), var_type);

	// Set the weights for label 0 as 1, label 1 as 10.
	// Make positive sample has more power in voting.
	vector<double> priors(2);
	priors[0] = 1;
	priors[1] = 10;

	cout << "Training the boost classifier (may take a few minutes)" << endl;

	Ptr<Boost> boost = Boost::create();
	boost->setBoostType(Boost::GENTLE);
	boost->setWeakCount(100);
	boost->setWeightTrimRate(0.95);
	boost->setMaxDepth(5);
	boost->setUseSurrogates(false);
	boost->setPriors(Mat(priors));

	boost->train(tdata);

	cout << "Boost training completed" << endl;

	return boost;
}

Mat_<float> extractFeatures(const vector<Mat> &digits) {
	Mat_<float> dataMat(digits.size(), 324);
	dataMat.setTo(0);

	cout << "Start extracting feature." << endl;

	for (int i = 0; i < digits.size(); i++) {
		HOGDescriptor *hog = new HOGDescriptor(Size(28, 28), Size(14, 14), Size(7, 7), Size(7, 7), 9);
		vector<float> descriptors;
		hog->compute(digits[i], descriptors, Size(1, 1), Size(0, 0));

		assert(descriptors.size() == 324);

		for (int j = 0; j < descriptors.size(); j++) {
			dataMat.at<float>(i, j) = descriptors[j];
		}

		delete hog;
	}

	cout << "Finish extracting." << endl;

	return dataMat;
}

vector<Mat> readDigits(string filePath) {
	ifstream file;
	vector<Mat> digits;

	// You must open ios_base::binary in Windows.
	file.open(filePath, ios_base::in | ios_base::binary);

	if (!file.is_open()) {
		cout << "File Not Found!" << endl;
		exit(2);
	}
	else {
		cout << "Start reading digits." << endl;

		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseDigit(magic_number);

		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseDigit(number_of_images);

		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseDigit(n_rows);

		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseDigit(n_cols);

		cout << "No. of images:" << number_of_images << endl;;

		Mat img;
		unsigned char *temp = new unsigned char[n_rows * n_cols];

		// Reading images.
		for (long int i = 0; i < number_of_images; ++i) {
			img.create(n_rows, n_cols, CV_8UC1);
	
			file.read((char*)temp, sizeof(unsigned char) * n_rows * n_cols);

			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					img.at<uchar>(r, c) = temp[r * n_rows + c];
				}
			}

			//imshow("img", img);
			//waitKey(0);

			digits.push_back(img);
			img.release();
		}
	}

	file.close();

	cout << "Finish reading digits." << endl;

	return digits;
}

Mat_<int> readLabels(string filePath) {
	int idx = 0;
	Mat_<int> labels;
	ifstream file;

	// You must open ios_base::binary in Windows.
	file.open(filePath, ios_base::in | ios_base::binary);

	if (!file.is_open()) {
		cout << "File Not Found!";
		exit(2);
	}
	else
	{
		cout << "Start reading labels." << endl;
		int magic_number = 0;
		int number_of_labels = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseDigit(magic_number);
		file.read((char*)&number_of_labels, sizeof(number_of_labels));
		number_of_labels = reverseDigit(number_of_labels);

		cout << "No. of labels:" << number_of_labels << endl;

		labels = Mat_<int>(number_of_labels, 1);

		for (long int i = 0; i < number_of_labels; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			labels.at<int>(i, 0) = (int)temp;
		}
	}

	file.close();

	cout << "Finish reading labels." << endl;

	return labels;
}

Mat_<int> getPredictLabels(Ptr<SVM> svm, const Mat_<float> &testMat) {
	assert(testMat.cols == 324);

	Mat_<int> predictLabels(testMat.rows, 1);

	for (int i = 0; i < testMat.rows; i++) {
		float res = svm->predict(testMat.row(i));
		predictLabels.at<int>(i, 0) = (int)res;
	}

	return predictLabels;
}

Mat_<int> getPredictLabels(Ptr<Boost> boost, const Mat_<float> &testMat) {
	int ntest_samples = testMat.rows;
	int var_count = testMat.cols;
	int class_count = 10;

	Mat_<int> predictLabels(ntest_samples, 1);
	Mat_<float> temp_sample(1, var_count + 1);
	float* tptr = temp_sample.ptr<float>();

	// compute prediction error on test data
	for (int i = 0; i < ntest_samples; i++) {

		const float* ptr = testMat.ptr<float>(i);
		for (int k = 0; k < var_count; k++) {
			tptr[k] = ptr[k];
		}

		int best_class = 0;
		double max_sum = -DBL_MAX;  // -DBL_MAX represents min value.
		for (int j = 0; j < class_count; j++) {
			tptr[var_count] = (float)j;
			float s = boost->predict(temp_sample, noArray(), StatModel::RAW_OUTPUT);

			if (max_sum < s) {
				max_sum = s;
				best_class = j;
			}
		}

		predictLabels.at<int>(i, 0) = best_class;
	}

	return predictLabels;
}

float getAccuracy(const Mat_<int> &testLabels, const Mat_<int> &predictLabels) {
	assert(testLabels.rows == predictLabels.rows);
	assert(testLabels.cols == predictLabels.cols);
	assert(predictLabels.cols == 1);

	float totalSum = testLabels.rows;
	float correctSum = 0;

	for (int i = 0; i < testLabels.rows; i++) {
		if (testLabels.at<int>(i, 0) == predictLabels.at<int>(i, 0)) {
			correctSum++;
		}
	}

	return correctSum / totalSum;
}

int reverseDigit(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int random(int min, int max) {
	assert(max > min);

	return rand() % (max - min + 1) + min;
}