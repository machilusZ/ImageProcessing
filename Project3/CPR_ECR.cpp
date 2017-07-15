#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <opencv2/core/core.hpp>  
#include "FFT_DCT.cpp"

using namespace std;
//using namespace cv;

namespace CPR_ECR {
	vector<vector<int> > compress(int quality, cv::Mat& image)
	{
		
		uchar * p;
		int rows = image.rows, cols = image.cols;
		vector<vector<int> > compressed_image(rows, vector<int>(cols));
		int blk_rows = rows / 8, blk_cols = cols / 8;

		for (int i = 0; i < blk_rows; i++) {
			for (int j = 0; j < blk_cols; j++) {
				int st_row = i * 8, st_col = j * 8;
				cv::Mat sub_img = image.rowRange(st_row, st_row + 8).colRange(st_col, st_col + 8);
				vector<vector<double> > sub = FFT_DCT::DCT(sub_img);

				for (int shift_row = 0; shift_row < 8; shift_row++) {
					for (int shift_col = 0; shift_col < 8; shift_col++) {
						compressed_image[st_row + shift_row][shift_col + st_col] = 
							(int)(0.5 + sub[shift_row][shift_col] / (1 + (1 + shift_col + shift_row) * quality));
					}	
				}
			}
		}
		return compressed_image;
	}

	vector<double> zigZag(vector<vector<double> > matrix)
	{
		int cols = matrix[0].size(), rows = matrix.size();
		int size = cols < rows ? rows : cols;
		vector<double> result; result.reserve(rows * cols);
		bool direction = true;
		for (int sum = 0; sum < size; sum++) {
			direction = !direction;
			if (direction) {
				for (int i = (sum - cols + 1 < 0 ? 0 : sum - cols + 1); i <= sum && i < rows; i++) {
					result.push_back(matrix[i][sum - i]);
				}
			}
			else {
				for (int j = (sum - rows + 1 < 0 ? 0 : sum - rows + 1); j <= sum && j < cols; j++) {
					result.push_back(matrix[sum - j][j]);
				}
			}
		}
		return result;
	}

	cv::Mat decompress(int quality, vector<vector<int> >& image)
	{
		//cv::Mat decompressed_image = image.clone();
		
		int rows = image.size(), cols = image[0].size();
		cv::Mat decompressed_image = cv::Mat::ones(rows, cols, CV_8UC1);
		vector<vector<double> > result(8, vector<double>(8));
		uchar * p;
		
		int blk_rows = rows / 8, blk_cols = cols / 8;

		for (int i = 0; i < blk_rows; i++) {
			for (int j = 0; j < blk_cols; j++) {
				int st_row = i * 8, st_col = j * 8;
				for (int shift_row = 0; shift_row < 8; shift_row++) {
					//p = (uchar *)image.ptr<uchar>(st_row + shift_row);
					for (int shift_col = 0; shift_col < 8; shift_col++) {
						result[shift_row][shift_col] = (image[st_row + shift_row][shift_col + st_col] * (1 + (1 + shift_col + shift_row) * quality));
					}

				}
				vector<vector<uchar> > sub_img = FFT_DCT::IDCT(result);
				for (int shift_row = 0; shift_row < 8; shift_row++) {
					p = (uchar *)decompressed_image.ptr<uchar>(st_row + shift_row);
					for (int shift_col = 0; shift_col < 8; shift_col++) {
						p[shift_col + st_col] = sub_img[shift_row][shift_col];
					}

				}
			}
		}
		return decompressed_image;
		//return result;
	}

	cv::Mat edge_extract_1(cv::Mat image)
	{
		cv::Mat result = image.clone();
		//Mat_<unsigned char> mat;
		uchar* p;
		//int theta[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		int theta_x[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
		int theta_y[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		int cols = result.cols, rows = result.rows;

		for (int i = 0; i < rows; i++) {
			p = (uchar*)result.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				int val_x = 0;
				int val_y = 0;
				for (int c = -1; c < 2; c++) {
					for (int r = -1; r < 2; r++) {
						if (i + r >= 0 && i + r < rows && j + c >= 0 && j + c < cols) {
							uchar* p = (uchar*)image.ptr<uchar>(i + r);
							val_x += p[j + c] * theta_x[4 + c + 3 * r];
							val_y += p[j + c] * theta_y[4 + c + 3 * r];
						}
					}
				}
				val_x = val_x < 0 ? -val_x : val_x;
				val_y = val_y < 0 ? -val_y : val_y;
				p[j] = val_x + val_y;
			}
		}
		return result;
	}

	cv::Mat edge_extract_2(cv::Mat image)
	{
		cv::Mat result = image.clone();
		//Mat_<unsigned char> mat;
		uchar* p;
		int theta[] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
		int cols = result.cols, rows = result.rows;

		for (int i = 0; i < rows; i++) {
			p = (uchar*)result.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				int val = 0;
				for (int c = -1; c < 2; c++) {
					for (int r = -1; r < 2; r++) {
						if (i + r >= 0 && i + r < rows && j + c >= 0 && j + c < cols) {
							uchar* p1 = (uchar*)image.ptr<uchar>(i + r);
							val += p1[j + c] * theta[4 + c + 3 * r];
						}
					}
				}

				p[j] = val < 0 ? -val : val;
			}
		}
		return result;
	}
	cv::Mat edge_extract_3(cv::Mat image, int radius, int mode)
	{
		cv::Mat result;
		vector<vector<FFT_DCT::Complex> > img_fft = FFT_DCT::FFT(FFT_DCT::imgProcess(image));
		//result = FFT_DCT::vec2Mat_M(FFT_DCT::IFFT(FFT_DCT::GHPF());
		switch (mode)
		{
		case(1): {
			result = FFT_DCT::vec2Mat_M(FFT_DCT::IFFT(FFT_DCT::GHPF(img_fft, radius)));
			break;
		}
		case(2): {
			result = FFT_DCT::vec2Mat_M(FFT_DCT::IFFT(FFT_DCT::BHPF(img_fft, radius)));
			break;
		}
		default:
			result = FFT_DCT::vec2Mat_M(FFT_DCT::IFFT(FFT_DCT::IHPF(img_fft, radius)));
			break;
		}
		//cv::Mat result = FFT_DCT::vec2Mat_M(FFT_DCT::IFFT((FFT_DCT::FFT(image))));
		return result;
	}

	void extream(cv::Mat image)
	{
		/*
		int max = 0, min = 256;
		uchar* p;
		for (int i = 0; i < image.rows; i++) {
			p = (uchar*)image.ptr<uchar>(i);
			for (int j = 0; j < image.cols; j++) {
				if ((int)p[j] < min)
					min = p[j];
				if ((int)p[j] > max)
					max = p[j];
			}
		}

		int mid = (max - min) / 5 + min;

		for (int i = 0; i < image.rows; i++) {
			p = (uchar*)image.ptr<uchar>(i);
			for (int j = 0; j < image.cols; j++) {
				if ((int)p[j] < mid)
					p[j] = 0;
				else
					p[j] = 255;
			}
		}
		*/
	}
}