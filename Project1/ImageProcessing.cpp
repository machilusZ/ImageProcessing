#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

using namespace std;
using namespace cv;
#define pic_round(x) ((x) > 255 ? 255 : (x) < 0 ? 0 : (x))

Mat histogram_equalization(Mat original_img)
{
	const int max_pixel = 256;
	

	Mat result = original_img.clone();  

	long n_rows = result.rows;
	long n_cols = result.cols * result.channels();
	long pixel_num = n_rows * n_cols;
	double CumuPixel[max_pixel]; memset(CumuPixel, 0, sizeof(double) * max_pixel);
	int NumPixel[max_pixel]; memset(NumPixel, 0, sizeof(int) * max_pixel);
	//double ProbPixel[max_pixel]; memset(ProbPixel, 0, sizeof(double) * max_pixel);
	/*
	* Row pointer
	*/
	uchar *p;
	for (long i = 0; i<n_rows; i++) {
		p = result.ptr<uchar>(i);
		for (long j = 0; j<n_cols; j++) {
			NumPixel[p[j]] += 1;
		}
	}

	CumuPixel[0] = (double)NumPixel[0] / pixel_num;

	for (int i = 1; i < max_pixel; i++) {
		CumuPixel[i] = CumuPixel[i - 1] + (double)NumPixel[i] / pixel_num;
	}

	for (long i = 0; i<n_rows; i++) {
		p = result.ptr<uchar>(i);
		for (long j = 0; j<n_cols; j++) {
			p[j] = (int)(CumuPixel[p[j]] * max_pixel + 0.5);
		}
	}

	return result;
}

Mat reverse_1channel(Mat original_image)
{
	Mat result = original_image.clone();  

	long n_rows = result.rows;
	long n_cols = result.cols * result.channels();

	uchar *p;
	for (long i = 0; i<n_rows; i++) {
		p = result.ptr<uchar>(i);
		for (long j = 0; j<n_cols; j++) {
			p[j] = 255 - p[j];
		}
	}
	return result;
}

Mat reverse(Mat original_image)
{
	if (original_image.channels() == 1)
		return reverse_1channel(original_image);
	else {
		vector<Mat> channels;
		split(original_image, channels);
		vector<Mat> result_channels;
		for (int i = 0; i < 3; i++) {
			result_channels.push_back(reverse_1channel(channels.at(i)));
		}
		Mat result; merge(result_channels, result);
		return result;
	}
}

Mat gray(Mat original_img)
{
	if (original_img.channels() == 1)
		return original_img.clone();
	Mat result; result.create(original_img.size(), CV_8UC1);

	long n_rows = result.rows;
	long n_cols = result.cols * result.channels();

	uchar* p = result.data;

	for (long i = 0; i<n_rows*n_cols; i+=1) {
		*(p + i) = (original_img.data[3 * i] + original_img.data[3 * i + 1] + original_img.data[3 * i + 2]) / 3;
	}

	return result;
}


Mat gray2(Mat original_img)
{
	Mat result; result.create(original_img.size(), original_img.type());

	long n_rows = result.rows;
	long n_cols = result.cols * result.channels();

	uchar* p = result.data;

	for (long i = 0; i<n_rows*result.cols; i += 1) {
		*(p + 3 * i) = (original_img.data[3 * i] + original_img.data[3 * i + 1] + original_img.data[3 * i + 2]) / 3;
		*(p + 3 * i + 1) = (original_img.data[3 * i] + original_img.data[3 * i + 1] + original_img.data[3 * i + 2]) / 3;
		*(p + 3 * i + 2) = (original_img.data[3 * i] + original_img.data[3 * i + 1] + original_img.data[3 * i + 2]) / 3;
	}

	return result;
}

Mat linear_1channel(Mat original_img, int max, int min, bool improve = false)
{
	Mat result = original_img.clone();
	if (max < min)
		return result;
	const int max_pixel = 256;
	double CumuPixel[max_pixel]; memset(CumuPixel, 0, sizeof(double) * max_pixel);
	long NumPixel[max_pixel]; memset(NumPixel, 0, sizeof(long) * max_pixel);
	long n_rows = result.rows;
	long n_cols = result.cols * result.channels();
	
	uchar *p;
	int ori_min = 256, ori_max = -1;

	for (long i = 0; i<n_rows; i++) {
		p = result.ptr<uchar>(i);
		for (long j = 0; j<n_cols; j++) {
			ori_max = ori_max >= p[j] ? ori_max : p[j];
			ori_min = ori_min <= p[j] ? ori_min : p[j];
			NumPixel[p[j]] += 1;
		}
	}
	if (improve) {
		double minp = 0.001, maxp = 0.98;
		bool set[2] = { false, false };
		long pixel_num = (n_rows * n_cols);
		CumuPixel[0] = (double)NumPixel[0] / pixel_num;
		if (CumuPixel[0] >= minp) {
			ori_min = 0;
			set[0] = true;
		}
		for (int i = 1; i < max_pixel; i++) {
			CumuPixel[i] = CumuPixel[i - 1] + (double)NumPixel[i] / pixel_num;
			if (CumuPixel[i] >= minp && !set[0]) {
				ori_min = i;
				set[0] = true;
			}
			else if (CumuPixel[i] > maxp && !set[1]) {
				ori_max = i - 1;
				set[1] = true;
			}
		}
		for (long i = 0; i<n_rows; i++) {
			p = result.ptr<uchar>(i);
			for (long j = 0; j<n_cols; j++) {
				if (p[j] <= ori_min)
					p[j] = min;
				else if (p[j] >= ori_max)
					p[j] = max;
				else
					p[j] = cvRound(((double)p[j] - ori_min) / (ori_max - ori_min) * (max - min) + min);
				p[j] = pic_round(p[j]);
			}
		}

	}
	else {
		for (long i = 0; i<n_rows; i++) {
			p = result.ptr<uchar>(i);
			for (long j = 0; j<n_cols; j++) {
				p[j] = cvRound(((double)p[j] - ori_min) / (ori_max - ori_min) * (max - min) + min);
				p[j] = pic_round(p[j]);
			}
		}
	}
	
	return result;
}

Mat linear(Mat original_img, vector<int> params, bool improve = false)
{
	if (original_img.channels() == 1)
		return linear_1channel(original_img, params.at(0), params.at(1), improve);
	else {
		vector<Mat> channels;
		split(original_img, channels);
		vector<Mat> result_channels;
		for (int i = 0; i < 3; i++) {
			result_channels.push_back(linear_1channel(channels.at(i), params.at(2 * i), params.at(2 * i + 1), improve));
		}
		Mat result; merge(result_channels, result);
		return result;
	}
}

Mat non_linear_1channel(Mat original_img, int mode, vector<double> params)
{
	Mat result = original_img.clone();
	long n_rows = result.rows;
	long n_cols = result.cols * result.channels();
	uchar* p;
	switch (mode)
	{
	case(0): {
		double a = params.at(0); double b = params.at(1); double lnc = log(params.at(2));
		printf("%lf %lf", log(100), log(params.at(2)));
		for (int i = 0; i < n_rows; i++) {
			p = result.ptr<uchar>(i);
			for (int j = 0; j < n_cols; j++) {
				p[j] = cvRound((double)a + log(p[j] + 1) / (b * lnc));
				
				p[j] = p[j] < 0 ? 0 :
					p[j] > 255 ? 255 : p[j];
			}
			
		}
		break;
	}
	case(1): {
		double a = params.at(0); double b = params.at(1); double c = params.at(2);
		
		for (int i = 0; i < n_rows; i++) {
			p = result.ptr<uchar>(i);
			for (int j = 0; j < n_cols; j++) {
				p[j] = cvRound(pow(b, c * (p[j] - a)) - 1);
				p[j] = p[j] < 0 ? 0 :
					p[j] > 255 ? 255 : p[j];
			}
		}
		break;
	}
	case(2): {
		double r = params.at(0);

		for (int i = 0; i < n_rows; i++) {
			p = result.ptr<uchar>(i);
			for (int j = 0; j < n_cols; j++) {
				p[j] = cvRound(pow(p[j], r));
				p[j] = p[j] < 0 ? 0 :
					p[j] > 255 ? 255 : p[j];
			}
		}
		break;
	}
	default:
		break;
	}
	return result;
}

Mat non_linear(Mat original_img, int mode, vector<double> params)
{
	if (original_img.channels() == 1)
		return non_linear_1channel(original_img, mode, params);
	else {
		vector<Mat> channels;
		split(original_img, channels);
		vector<Mat> result_channels;
		for (int i = 0; i < 3; i++) {
			result_channels.push_back(non_linear_1channel(channels.at(i), mode, params));
		}
		Mat result; merge(result_channels, result);
		return result;
	}
}

vector<double> calcCumuPorb(Mat img) {
	const int max_pixel = 256;
	double CumuPixel[max_pixel]; memset(CumuPixel, 0, sizeof(double) * max_pixel);
	int NumPixel[max_pixel]; memset(NumPixel, 0, sizeof(int) * max_pixel);
	long n_rows = img.rows;
	long n_cols = img.cols * img.channels();
	//double ProbPixel[max_pixel]; memset(ProbPixel, 0, sizeof(double) * max_pixel);
	/*
	* Row pointer
	*/
	uchar *p;
	for (long i = 0; i<n_rows; i++) {
		p = img.ptr<uchar>(i);
		for (long j = 0; j<n_cols; j++) {
			NumPixel[p[j]] += 1;
		}
	}
	long pixel_num = n_rows * n_cols;

	CumuPixel[0] = (double)NumPixel[0] / pixel_num;
	double max = 0;
	for (int i = 1; i < max_pixel; i++) {
		CumuPixel[i] = CumuPixel[i - 1] + (double)NumPixel[i] / pixel_num;
		max = CumuPixel[i] - CumuPixel[i - 1] > max ? CumuPixel[i] - CumuPixel[i - 1] : max;
	}
	vector<double> vecHeight(CumuPixel, CumuPixel + sizeof(CumuPixel) / sizeof(double));
	vecHeight.push_back(max);
	return vecHeight;
}

Mat create_hist_image_gray(Mat img)
{
	int bins = 256;
	vector<double> cumu_prob = calcCumuPorb(img);
	int hist_height = 256;
	double max_val;
	int scale = 2;
	

	Mat hist_img = Mat::zeros(hist_height, bins*scale, CV_8UC3);

	double sum = 0;
	for (int i = 0; i<bins; i++)
	{
		double bin_val = cumu_prob.at(i) - (i == 0 ? 0 : cumu_prob.at(i - 1));
		sum += bin_val;
		
		int intensity = cvRound(bin_val / cumu_prob.at(bins)*hist_height);

		rectangle(hist_img, Point(i*scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
	
	return hist_img;

}

Mat create_hist_image_color(Mat img)
{
	vector<Mat> rgb_planes;
	split(img, rgb_planes);

	/// 设定bin数目
	int histSize = 255;

	/// 设定取值范围 ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat r_hist, g_hist, b_hist;

	/// 计算直方图:
	calcHist(&rgb_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// 创建直方图画布
	int hist_w = 400; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

	/// 将直方图归一化到范围 [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// 在直方图画布上画出直方图
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImage;
}

Mat create_hist_image(Mat img)
{
	if (img.channels() == 1)
		return create_hist_image_gray(img);
	else
		return create_hist_image_color(img);
}

int main(int argc, char *argv[])
{
	char a[10000] = "lena512.bmp\0";
	cin.getline(a, 10000);
	

	Mat img;

	img = imread(a, -1);
	//printf("%d", img.channels());
	Mat out1 = create_hist_image(img);
	
	/*
	* Display
	*/
	namedWindow("src_image", CV_WINDOW_NORMAL);
	imshow("src_image", img);
	namedWindow("src_hist", CV_WINDOW_NORMAL);
	imshow("src_hist", out1);
	waitKey(0);
	char m;
	String imageName = "", histName;

	printf("Input Operation Code:\n");

	while (scanf(" %c", &m) > 0 && m != 'q') {
		Mat result, hist;
		if (imageName != "") {
			/*if (m == 's') {
				imwrite("E:/g.jpg", result);
				waitKey(0);
				printf("Input Operation Code:\n");
				continue;
			}*/
			destroyWindow(imageName);
			destroyWindow(histName);
			imageName = "";
			histName = "";
			
		}
		switch (m)
		{
		case('r'): {
			result = reverse(img);
			imageName = "reverse_image";
			histName = "reverse_hist";
			break;
		}
		case('g'): {
			result = gray(img);
			imageName = "gray_image";
			histName = "gray_hist";
			break;
		}
		case('l'): {
			vector<int> params;
			if (img.channels() == 1) {
				printf("input lowwer and upper bounds:\n");
				double c, d; scanf("%lf %lf", &c, &d);
				params.push_back(d); params.push_back(c);
			}
			else {
				char* channels[] = { "for channel B", "for channel G", "for channel R" };
				for (int i = 0; i < img.channels(); i++) {
					printf("input lowwer and upper bounds %s:\n", channels[i]);
					double c, d; scanf("%lf %lf", &c, &d);
					params.push_back(d); params.push_back(c);
				}
			}
			printf("choose mode(0 or 1 or 2 or 3):\n");
			int mode; scanf("%d", &mode);
			if (mode >= 2) {
				result = histogram_equalization(img);
				result = linear(result, params, mode - 2);
			}
			else {
				result = linear(img, params, mode);
			}
			imageName = "linear_image";
			histName = "linear_hist";
			break;
		}
		case('n'): {
			printf("input mode(p or m or d):\n");
			char mode; scanf(" %c", &mode);
			switch (mode)
			{
			case('d'): {
				printf("input the 3 params:\n");
				double a, b, c;
				scanf("%lf %lf %lf", &a, &b, &c);
				result = non_linear(img, 0, { a, b, c });
				imageName = "log_image";
				histName = "log_hist";
				break;
			}
			case('m'): {
				printf("input the 3 params:\n");
				double a, b, c;
				scanf("%lf %lf %lf", &a, &b, &c);
				result = non_linear(img, 1, { a, b, c });
				imageName = "power_image";
				histName = "power_hist";
				break;
			}
			case('p'): {
				printf("input the param:\n");
				double r;
				scanf("%lf", &r);
				result = non_linear(img, 2, { r });
				imageName = "power_image";
				histName = "power_hist";
				break;
			}
			default:
				break;
			}
			break;
		}
		case('h'): {
			result = histogram_equalization(img);
			imageName = "hist_equ_image";
			histName = "hist_equ_hist";
			break;
		}
		default:
			printf("Invalid Operation!\n");
			printf("q g l n h\n");
		}
		if (imageName != "") {
			hist = create_hist_image(result);
			//imwrite("E:/g.jpg", result);
			namedWindow(imageName, CV_WINDOW_NORMAL);
			imshow(imageName, result);
			namedWindow(histName, CV_WINDOW_NORMAL);
			imshow(histName, hist);
			waitKey(0);
		}
		
		printf("Input Operation Code:\n");
	}
	
	
	return 0;
}