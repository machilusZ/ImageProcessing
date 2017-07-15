// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"


#include <math.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>

#include "FFT_DCT.cpp"



using namespace std;
using namespace cv;

int cmp(const void* a, const void* b)
{
	return *(double*)a > *(double*)b ? 1 : -1;
}


void normalize(Mat original_mat, double max = 1.0)
{
	double minVal = 0, maxVal = 0;
	// Localize minimum and maximum values  
	//cvMinMaxLoc(original_mat, &minVal, &maxVal);
	double* p;
	for (int i = 0; i < original_mat.rows; i++) {
		p = original_mat.ptr<double>(i);
		for (int j = 0; j < original_mat.cols; j++)
			p[j] = log(1 + abs(p[j]));
	}


	minMaxIdx(original_mat, &minVal, &maxVal);
	// Normalize image (0 - 255) to be observed as an u8 image  
	double scale = max / (maxVal - minVal);
	double shift = -minVal * scale;

	
	
	for (int i = 0; i < original_mat.rows; i++) {
		p = original_mat.ptr<double>(i);
		for (int j = 0; j < original_mat.cols; j++)
			p[j] = p[j] * scale + shift;
	}


	
	//cvConvertScale(original_mat, original_mat, scale, shift);
	//cvCon
}

Mat resize(Mat original)
{
	Size dsize = Size(FFT_DCT::roundup2(original.cols), FFT_DCT::roundup2(original.rows));
	Mat result = Mat(dsize, CV_64FC1);
	//result.create(, CV_64FC1);
	resize(original, result, dsize);
	return result;
}

bool check(Mat unchecked_img)
{
	int cols = unchecked_img.cols, rows = unchecked_img.rows;
	double* p;
	for (int i = 0; i < rows; i++) {
		p = unchecked_img.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			if (p[j] < 0)
				return false;
		}
	}
	return true;
}

int main()
{
	char a[10000] = "C:\\Users\\70478\\Desktop\\fy\\图像处理\\Lena\\lena512.bmp";
	printf("请输入图片名\n");

	cin.getline(a, 10000);
	
	Mat img = imread(a, -1);
	if(img.channels() > 1)
		cvtColor(img, img, CV_BGR2GRAY);

	img.convertTo(img, CV_64FC1, 1 / 255.0);
	img = resize(img);
	int cols = img.cols, rows = img.rows;
	cout << cols << endl;
	vector<vector<FFT_DCT::Complex> > result_fft;
	
	Mat mM;


	char m;
	printf("FFT(f) or DCT(d)?\n");
	scanf_s(" %c", &m);

	namedWindow("原图", CV_WINDOW_NORMAL);
	imshow("原图", img);

	if (m == 'f') {
		result_fft = FFT_DCT::FFT(FFT_DCT::imgProcess(img));
		mM = FFT_DCT::vec2Mat_M(result_fft);
		normalize(mM);
		namedWindow("原图频谱", CV_WINDOW_NORMAL);
		imshow("原图频谱", mM);
		waitKey(0);
	}
	else {
		vector<vector<double> > result_dct = FFT_DCT::DCT((img));
		vector<vector<double> > result_idct = FFT_DCT::IDCT(result_dct);
		Mat mID = FFT_DCT::vec2Mat((result_idct));
		namedWindow("DCT逆变换", CV_WINDOW_NORMAL);
		imshow("DCT逆变换", mID);
		Mat mD = FFT_DCT::vec2Mat(result_dct);
		normalize(mD, 4);
		cv::normalize(mD, mD, 20, 0, NORM_MINMAX);
		/*double *p;
		for (int i = 0; i < mD.rows; i++) {
			p = mD.ptr<double>(i);
			for (int j = 0; j < mD.cols; j++)
				p[j] *= 1000;
		}*/
		//FFT_DCT::fftshift(mD);
		namedWindow("原图频谱", CV_WINDOW_NORMAL);
		imshow("原图频谱", mD);
		waitKey(0);
		return 0;
	}
	
	String imageName = "";
	Mat result, fre;

	printf("Input operation code\n");

	while (scanf_s(" %c", &m) > 0 && m != 'q') {
		if (imageName != "") {
			destroyWindow(imageName);
			destroyWindow(imageName + "频谱");
			imageName = "";
		}
		

		switch (m) {
		case('h'): {
			int radius;
			double a, b;
			printf("Input HPF mode && radius && a, b\n");
			scanf_s(" %c %d %lf %lf", &m, 1, &radius, &a, &b);
			vector<vector<FFT_DCT::Complex> > temp_fre;
			if (m == 'i') {
				temp_fre = FFT_DCT::IHPF(result_fft, radius, a, b);
				imageName = "IHPF";
			}
			else if (m == 'b') {
				printf("level?\n");
				int level;
				scanf_s("%d", &level);
				temp_fre = FFT_DCT::BHPF(result_fft, radius, a, b, level);
				imageName = "BHPF";
			}
			else {
				temp_fre = FFT_DCT::GHPF(result_fft, radius, a, b);
				imageName = "GHPF";
			}
			fre = FFT_DCT::vec2Mat_M(temp_fre);
			result = FFT_DCT::vec2Mat_M(FFT_DCT::IFFT(temp_fre));
			break;
		}
		case('l'): {
			int radius = 20;
			printf("Input LPF mode && radius\n");
			scanf_s(" %c %d", &m, 1, &radius);
			
			vector<vector<FFT_DCT::Complex> > temp_fre;
			if (m == 'i') {
				temp_fre = FFT_DCT::ILPF(result_fft, radius);
				imageName = "ILPF";
			}
			else if (m == 'b') {
				printf("level?\n");
				int level;
				scanf_s("%d", &level);
				temp_fre = FFT_DCT::BLPF(result_fft, radius, level);
				imageName = "BLPF";
			}
			else {
				temp_fre = FFT_DCT::GLPF(result_fft, radius);
				imageName = "GLPF";
			}
			fre = FFT_DCT::vec2Mat_M(temp_fre);
			result = FFT_DCT::vec2Mat_M(FFT_DCT::IFFT(temp_fre));
			break;
		}
		default: {
			printf("Invalid Operation!\n");
			printf("h l q\n");
		}
		}
		if (imageName != "") {
			namedWindow(imageName, CV_WINDOW_NORMAL);
			imshow(imageName, result);
			normalize(fre);
			namedWindow(imageName + "频谱", CV_WINDOW_NORMAL);
			imshow(imageName + "频谱", fre);
			waitKey(0);
		}
		printf("Input operation code\n");
	}


	return 0;
}

