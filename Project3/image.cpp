// ConsoleApplication1.cpp : ¶¨Òå¿ØÖÆÌ¨Ó¦ÓÃ³ÌÐòµÄÈë¿Úµã¡£
//
#include "stdafx.h"


#include <math.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>

//#include "FFT_DCT.cpp"
#include "CPR_ECR.cpp"



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
	Mat result = Mat(dsize, CV_8UC1);
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
	char a[10000] = "";
	//printf("ÇëÊäÈëÍ¼Æ¬Ãû\n");
	printf("请输入文件路径\n");
	cin.getline(a, 10000);

	Mat img = imread(a, -1);
	if (img.channels() > 1) {
		cvtColor(img, img, CV_BGR2GRAY);
	}
	//img.convertTo(img, CV_16UC1, 1);
	Mat img_double;
	img = resize(img);
	img.convertTo(img_double, CV_64FC1, 1 / 255.0);
	
	int cols = img.cols, rows = img.rows;
	//cout << cols << endl;
	
	printf("Input operation code\n");
	
	string imageName = "";
	char m;
	Mat result;

	
	while (scanf_s("%c", &m, &m) > 0 && m != 'q') {
		
		if (imageName != "") {
			destroyWindow(imageName);
			destroyWindow("original");
			imageName = "";
		}


		switch (m) {
		case('c'): {
			imageName = "Compressed image";
			printf("Please input the compression rate\n");
			int r;
			scanf_s("%d", &r);
			result = CPR_ECR::decompress(r, CPR_ECR::compress(r, img));
			break;
		}
		case('e'): {
			imageName = "Edge Extraction";
			printf("Please input the extraction mode:(1 or 2 or 3)\n");
			int mode = 0;
			scanf_s("%d", &mode);
			mode = mode % 3 + 1;
			if (mode == 1) {
				printf("input the radius\n");
				int radius; scanf_s("%d", &radius);
				printf("input the filter mode(1 or 2 or 3)\n");
				scanf_s("%d", &mode);
				result = CPR_ECR::edge_extract_3(img_double, radius, mode);
			}
			else if (mode == 3){
				result = CPR_ECR::edge_extract_2(img);
			}
			else {
				result = CPR_ECR::edge_extract_1(img);
			}
			CPR_ECR::extream(result);
			break;
		}
		default: {
			printf("Invalid Operation!\n");
			printf("c e q\n");
		}
		}
		if (imageName != "") {
			namedWindow(imageName, CV_WINDOW_NORMAL);
			imshow(imageName, result);
			
			namedWindow("original", CV_WINDOW_NORMAL);
			imshow("original", img);
			waitKey(0);
		}
		printf("Input operation code\n");
	}


	return 0;
}

