#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <opencv2/core/core.hpp>  

#define PI 3.14159265

using namespace std;
//using namespace cv;
namespace FFT_DCT {
	class Complex
	{
	public:
		double image;
		double real;
		Complex() {

		}
		Complex(double real, double image) {
			this->real = real;
			this->image = image;
		}
		Complex operator *(const Complex& c) {
			return Complex(this->real * c.real - this->image * c.image, this->real * c.image + this->image * c.real);
		}
		Complex operator *(double v) {
			return Complex(this->real * v, this->image * v);
		}
		Complex operator +(const Complex& c) {
			return Complex(this->real + c.real, this->image + c.image);
		}
		Complex operator +(double v) {
			return Complex(this->real + v, this->image);
		}
		Complex operator -(const Complex& c) {
			return Complex(this->real - c.real, this->image - c.image);
		}
		Complex operator /(double N) {
			return Complex(this->real / N, this->image / N);
		}
	};

	ostream& operator <<(ostream& out, const Complex& c)
	{
		if (c.image > 0) {
			cout << c.real << "+" << c.image << "i";
		}
		else if (c.image < 0) {
			cout << c.real << "-" << -c.image << "i";
		}
		else {
			cout << c.real;
		}
		return out;
	}

	ostream& operator <<(ostream& out, const vector<double>& v)
	{
		for (int i = 0; i < v.size(); i++)
			out << v.at(i) << " ";
		out << endl;
		return out;
	}

	ostream& operator <<(ostream& out, const vector<Complex>& v)
	{
		for (int i = 0; i < v.size(); i++)
			out << v.at(i) << " ";
		out << endl;
		return out;
	}

	long reverse(long x, int length)
	{
		if (!length) {
			return 0;
		}
		else if (length == 1) {
			return x;
		}
		int half = length / 2;
		long lowwer = x & ((1 << half) - 1);
		return (reverse(lowwer, half) << (length - half)) + (reverse((x - lowwer) >> half, length - half));
	}

	int roundup2(int val) {
		{
			if (val & (val - 1)) {
				val |= val >> 1;
				val |= val >> 2;
				val |= val >> 4;
				val |= val >> 8;
				val |= val >> 16;
				return val + 1;
			}
			else {
				return val == 0 ? 1 : val;
			}
		}
	}

	int len(int N)
	{
		int length = 0;
		while (N > 0) {
			length += 1;
			N = (N >> 1);
		}
		return length - 1;
	}

	vector<Complex> FFT_1(double x[], int N)
	{
		vector<double> x_v(x, x + N);
		int v = roundup2(N);
		int N_old = N;
		while (v != N) {
			N += 1;
			x_v.push_back(0);
		}
		int length = len(N);


		vector<Complex> W(N / 2);

		Complex value = Complex(cos(2.0 * PI / N), -sin(2.0 * PI / N));
		W[0] = Complex(1, 0);
		for (int i = 1; i < N / 2; i++)
			W[i] = W[i - 1] * value;

		//Complex* values = new Complex[N];
		vector<Complex> values(N);

		for (int i = 0; i < N; i++) {
			values[i].real = x_v[reverse((long)i, length)];
			values[i].image = 0;
		}

		Complex temp;
		int interval = 1;

		while (interval <= N / 2) {

			int size = (interval << 1);
			for (int i = 0; i < N; i += size) {
				for (int j = 0; j < interval; j++) {
					temp = values[i + j + interval] * W[N * j / size];
					values[i + j + interval] = values[i + j] - temp;
					values[i + j] = values[i + j] + temp;
				}
			}

			interval = size;
		}

		while (N > N_old) {
			values.pop_back();
			N--;
		}
		return values;
	}


	vector<Complex> FFT_1(vector<Complex> x)
	{
		int N = x.size();
		int v = roundup2(N);
		int N_old = N;

		
		while (N < v) {
			x.push_back(Complex(0,0));
			N++;
		}
		

		int length = len(N);

		//Complex* W = new Complex[N / 2];
		vector<Complex> W(N / 2);
		Complex value = Complex(cos(2.0 * PI / N), -sin(2.0 * PI / N));
		W[0] = Complex(1, 0);
		for (int i = 1; i < N / 2; i++)
			W[i] = W[i - 1] * value;

		//Complex* values = new Complex[N];
		vector<Complex> values(N);

		for (int i = 0; i < N; i++) {
			// values[i].real = x[reverse(i, length)];
			// values[i].real = x.at(reverse(i, length)).real;
			// values[i].image = 0;
			values[i] = x[reverse((long)i, length)];
		}

		Complex temp;
		int interval = 1;

		while (interval <= N / 2) {

			int size = (interval << 1);
			for (int i = 0; i < N; i += size) {
				for (int j = 0; j < interval; j++) {
					temp = values[i + j + interval] * W[N * j / size];
					values[i + j + interval] = values[i + j] - temp;
					values[i + j] = values[i + j] + temp;
				}
			}

			interval = size;
		}

		while (N > N_old) {
			values.pop_back();
			N--;
		}
		//vector<Complex> result(values, values + N_old);
		return values;
	}

	vector<Complex> FFT_1(Complex x[], int N)
	{
		vector<Complex> value(x, x + N);
		return FFT_1(value);
	}

	/*vector<Complex> FFT_1(cv::Mat m)
	{
		
		return value;
	}*/
	vector<Complex> IFFT_1(double x[], int N)
	{
		vector<double> x_v(x, x + N);
		int v = roundup2(N);
		int N_old = N;
		while (v != N) {
			N += 1;
			x_v.push_back(0);
		}
		int length = len(N);


		vector<Complex> W(N / 2);

		Complex value = Complex(cos(2.0 * PI / N), sin(2.0 * PI / N));
		W[0] = Complex(1, 0);
		for (int i = 1; i < N / 2; i++)
			W[i] = W[i - 1] * value;

		//Complex* values = new Complex[N];
		vector<Complex> values(N);

		for (int i = 0; i < N; i++) {
			values[i].real = x_v[reverse((long)i, length)];
			values[i].image = 0;
		}

		Complex temp;
		int interval = 1;

		while (interval <= N / 2) {

			int size = (interval << 1);
			for (int i = 0; i < N; i += size) {
				for (int j = 0; j < interval; j++) {
					temp = values[i + j + interval] * W[N * j / size];
					values[i + j + interval] = values[i + j] - temp;
					values[i + j] = values[i + j] + temp;
				}
			}

			interval = size;
		}

		for (int i = 0; i < N_old; i++)
			values[i] = values[i] / N_old;
		while (N > N_old) {
			values.pop_back();
			N--;
		}
		return values;
	}


	vector<Complex> IFFT_1(vector<Complex> x)
	{
		int N = x.size();
		
		int v = roundup2(N);
		int N_old = N;


		while (N < v) {
			x.push_back(Complex(0, 0));
			N++;
		}
		int length = len(N);

		vector<Complex> W(N / 2);
		Complex value = Complex(cos(2.0 * PI / N), sin(2.0 * PI / N));
		W[0] = Complex(1, 0);
		for (int i = 1; i < N / 2; i++)
			W[i] = W[i - 1] * value;

		vector<Complex> values(N);

		for (int i = 0; i < N; i++) {
			// values[i].real = x[reverse(i, length)];
			// values[i].real = x.at(reverse(i, length)).real;
			// values[i].image = 0;
			values[i] = x[reverse((long)i, length)];
		}

		Complex temp;
		int interval = 1;

		while (interval <= N / 2) {

			int size = (interval << 1);
			for (int i = 0; i < N; i += size) {
				for (int j = 0; j < interval; j++) {
					temp = values[i + j + interval] * W[N * j / size];
					values[i + j + interval] = values[i + j] - temp;
					values[i + j] = values[i + j] + temp;
				}
			}

			interval = size;
		}

		for (int i = 0; i < N_old; i++)
			values[i] = values[i] / N_old;

		while (N > N_old) {
			values.pop_back();
			N--;
		}
		//vector<Complex> result(values, values + N_old);
		return values;
	}

	vector<Complex> IFFT_1(Complex x[], int N)
	{
		vector<Complex> value(x, x + N);
		return IFFT_1(value);
	}

	
	vector<vector<double> > reverse(double* mat, int rows, int cols)
	{

		double temp;

		vector<vector<double> > r(cols, vector<double>(rows));

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				r[j][i] = mat[i * cols + j];
			}
		}

		return r;
	}

	vector<vector<Complex> > reverse(vector<vector<Complex> > mat)
	{

		Complex temp;
		int cols = (mat[0]).size(), rows = mat.size();
		vector<vector<Complex> > r(cols, vector<Complex>(rows));

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				r[j][i] = mat[i][j];
			}
		}

		return r;
	}

	vector<vector<double> > reverse(vector<vector<double> > mat)
	{

		Complex temp;
		int cols = (mat[0]).size(), rows = mat.size();
		vector<vector<double> > r(cols, vector<double>(rows));

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				r[j][i] = mat[i][j];
			}
		}

		return r;
	}

	vector<vector<Complex> > FFT(double * f, int rows, int cols)
	{

		// vector<vector<Complex> values;
		vector<vector<double> > r = reverse(f, rows, cols);
		//Complex values[N][N];
		vector<vector<Complex> > values(rows, vector<Complex>(cols));
		vector<vector<Complex> > results;

		for (int i = 0; i < cols; i++) {
			vector<Complex> value = FFT_1(&r.at(i)[0], rows);
			for (int j = 0; j < rows; j++)
				values[j][i] = value.at(j);
		}

		for (int i = 0; i < rows; i++) {
			results.push_back(FFT_1(values[i]));
		}
		return results;
	}

	vector<vector<Complex> > FFT(cv::Mat m)
	{
		int rows = m.rows; int cols = m.cols;
		cv::Mat r = m.t();
		
		vector<vector<Complex> > values(rows, vector<Complex>(cols));
		vector<vector<Complex> > results;

		for (int i = 0; i < cols; i++) {
			vector<Complex> value = FFT_1(r.ptr<double>(i), rows);
			for (int j = 0; j < rows; j++)
				values[j][i] = value.at(j);
		}

		for (int i = 0; i < rows; i++) {
			results.push_back(FFT_1(values[i]));
		}
		return results;
	}

	vector<double> DCT_1(double x[], int N)
	{
		//double C[N];
		//double x_e[2 * N];
		vector<double> C(N);
		vector<double> x_e(2 * N);
		memset(&x_e[0], 0, sizeof(double) * (2 * N));
		memcpy(&x_e[0], x, sizeof(double) * N);
		double N_sqrt = sqrt((double)N);
		double N_2_sqrt = sqrt(2.0 / N);

		//Complex W[N];
		vector<Complex> W(N);
		Complex value = Complex(cos(PI / (2.0 * N)), -sin(PI / (2.0 * N)));
		W[0] = Complex(1, 0);
		for (int i = 1; i < N; i++)
			W[i] = W[i - 1] * value;

		C[0] = 0;
		for (int i = 0; i < N; i++) {
			C[0] += x[i];
		}
		C[0] /= N_sqrt;

		vector<Complex> fft_result = FFT_1(&x_e[0], 2 * N);
		for (int i = 1; i < N; i++) {
			C[i] = N_2_sqrt * (W[i] * fft_result.at(i)).real;
		}

		//vector<double> result(C, C + N);
		return C;
	}

	vector<double> IDCT_1(double x[], int N)
	{
		//double C[N];
		//double x_e[2 * N];
		vector<double> C(N);
		vector<double> x_e(2 * N);
		memset(&x_e[0], 0, sizeof(double) * (2 * N));
		memcpy(&x_e[0], x, sizeof(double) * N);
		double N_sqrt = sqrt(1.0 / N);
		double N_2_sqrt = sqrt(2.0 / N);

		//Complex W[N];
		vector<Complex> W(2 * N);
		Complex value = Complex(cos(PI / (2.0 * N)), sin(PI / (2.0 * N)));
		W[0] = Complex(1, 0);
		for (int i = 1; i < 2 * N; i++)
			W[i] = W[i - 1] * value;

		vector<Complex> V(2 * N); for (int i = 0; i < 2 * N; i++) V[i] = W[i] * x_e[i];


		/*C[0] = 0;
		for (int i = 0; i < N; i++) {
			C[0] += x[i];
		}
		C[0] /= N_sqrt;*/

		vector<Complex> fft_result = IFFT_1(V);
		for (int i = 0; i < N; i++) {
			C[i] = 2 * N * N_2_sqrt * fft_result.at(i).real + (N_sqrt - N_2_sqrt) * x[0];
		}

		//vector<double> result(C, C + N);
		return C;
	}

	vector<vector<double> > DCT(double * f, int rows, int cols)
	{

		vector<vector<double> > r = reverse(f, rows, cols);
		//Complex values[N][N];
		vector<vector<double> > values(rows, vector<double>(cols));
		vector<vector<double> > results;

		for (int i = 0; i < cols; i++) {
			vector<double> value = DCT_1(&r.at(i)[0], rows);
			for (int j = 0; j < rows; j++)
				values[j][i] = value.at(j);
		}

		for (int i = 0; i < rows; i++) {
			results.push_back(DCT_1(&(values[i])[0], cols));
		}
		return results;
	}

	
	vector<vector<double> > DCT(cv::Mat m)
	{
		int cols = m.cols; int rows = m.rows;
		//vector<vector<double> > r = reverse(f, rows, cols);
		cv::Mat r = m.t();
		//Complex values[N][N];
		vector<vector<double> > values(rows, vector<double>(cols));
		vector<vector<double> > results;

		for (int i = 0; i < cols; i++) {
			vector<double> value = DCT_1(r.ptr<double>(i), rows);
			for (int j = 0; j < rows; j++)
				values[j][i] = value.at(j);
		}

		for (int i = 0; i < rows; i++) {
			results.push_back(DCT_1(&(values[i])[0], cols));
		}
		return results;
	}


	vector<vector<double> > IDCT(vector<vector<double> > f)
	{
		int rows = f.size(), cols = f[0].size();
		vector<vector<double> > r = reverse(f);
		//Complex values[N][N];
		vector<vector<double> > values(rows, vector<double>(cols));
		vector<vector<double> > results;

		for (int i = 0; i < cols; i++) {
			vector<double> value = IDCT_1(&r.at(i)[0], rows);
			for (int j = 0; j < rows; j++)
				values[j][i] = value.at(j);
		}

		for (int i = 0; i < rows; i++) {
			results.push_back(IDCT_1(&(values[i])[0], cols));
		}
		return results;
	}

	vector<vector<double> > IDCT(cv::Mat m)
	{
		int cols = m.cols; int rows = m.rows;
		//vector<vector<double> > r = reverse(f, rows, cols);
		cv::Mat r = m.t();
		//Complex values[N][N];
		vector<vector<double> > values(rows, vector<double>(cols));
		vector<vector<double> > results;

		for (int i = 0; i < cols; i++) {
			vector<double> value = IDCT_1(r.ptr<double>(i), rows);
			for (int j = 0; j < rows; j++)
				values[j][i] = value.at(j);
		}

		for (int i = 0; i < rows; i++) {
			results.push_back(IDCT_1(&(values[i])[0], cols));
		}
		return results;
	}

	vector<vector<Complex> > IFFT(vector<vector<Complex> > f)
	{
		vector<vector<Complex> > r = reverse(f);
		//Complex values[N][N];
		int rows = f.size(), cols = (f[0]).size();
		vector<vector<Complex> > values(rows, vector<Complex>(cols));
		vector<vector<Complex> > results;

		for (int i = 0; i < cols; i++) {
			vector<Complex> value = IFFT_1(&r.at(i)[0], rows);
			for (int j = 0; j < rows; j++)
				values[j][i] = value.at(j);
		}

		for (int i = 0; i < rows; i++) {
			results.push_back(IFFT_1(values[i]));
		}
		return results;
	}

	cv::Mat vec2Mat_M(vector<vector<Complex> > v)
	{
		cv::Mat result(v.size(), v[0].size(), CV_64FC1);

		for (int i = 0; i < v.size(); i++) {
			vector<double> value(v[i].size());
			for (int j = 0; j < v[i].size(); j++) {
				value[j] = sqrt(v[i][j].real * v[i][j].real + v[i][j].image * v[i][j].image);
			}
			cv::Mat row = cv::Mat(1, value.size(), CV_64FC1, (double *)value.data());
			cv::Mat p = result.row(i);
			row.copyTo(p);
		}
		
		return result;
	}

	cv::Mat vec2Mat_F(vector<vector<Complex> > v)
	{
		cv::Mat result(v.size(), v[0].size(), CV_64FC1);
		for (int i = 0; i < v.size(); i++) {
			vector<double> value(v[i].size());
			for (int j = 0; j < v[i].size(); j++) {
				value[j] = atan2(v[i][j].image, v[i][j].real);
			}
			cv::Mat row = cv::Mat(1, value.size(), CV_64FC1, (double *)value.data());
			cv::Mat p = result.row(i);
			row.copyTo(p);
		}

		return result;
	}

	cv::Mat vec2Mat(vector<vector<double> > v) 
	{
		cv::Mat result(v.size(), v[0].size(), CV_64FC1);
		for (int i = 0; i < v.size(); i++) {
			cv::Mat row = cv::Mat(1, v[i].size(), CV_64FC1, (double *)v[i].data());
			cv::Mat p = result.row(i);
			row.copyTo(p);
		}

		return result;
	}

	void fftshift(cv::Mat img)
	{
		int cols = img.cols / 2, rows = img.rows / 2;
		cv::Mat_<double> im = img;
		double tmp13, tmp24;

		for (int j = 0; j < rows; j++) {
			for (int i = 0; i < cols; i++) {
				tmp13 = im(j, i);
				im(j, i) = im(j + rows, i + cols);
				im(j + rows, i + cols) = tmp13;
				
				tmp24 = im(j, i + cols);
				im(j, i + cols) = im(j + rows, i);
				im(j + rows, i) = tmp24;
			}
		}
	}

	cv::Mat imgProcess(cv::Mat original_img)
	{
		int cols = original_img.cols, rows = original_img.rows;
		cv::Mat result = original_img.clone();
		double* p;
		for (int i = 0; i < rows; i++) {
			p = result.ptr<double>(i);
			int i_2 = i % 2;
			for (int j = 0; j < cols; j++)
				p[j] = (j % 2 + i_2) % 2 ? -p[j] : p[j];
		}

		return result;
	}





	/*vector<vector<Complex> > copy(vector<vector<Complex> > src)
	{
		vector<vector<Complex> > values(src.size(), vector<Complex>(src[0].size()));

		for (int i = 0; i < src.size(); i++) {
			for (int j = 0; j < src[0].size(); j++) {
				values[i][j] = src[i][j];
			}
		}

		return values;
	}*/

	vector<vector<Complex> > ILPF(vector<vector<Complex> >& original_mat, int radius)
	{
		int cols = original_mat[0].size(), rows = original_mat.size();
		int center_x = (cols - 1) / 2, center_y = (rows - 1) / 2;
		int radius2 = radius * radius;
		vector<vector<Complex> > result(original_mat.size(), vector<Complex>(original_mat[0].size()));


		for (int i = 0; i < rows; i++) {
			int x2 = (i - center_x) * (i - center_x);
			for (int j = 0; j < cols; j++) {
				int y2 = (j - center_y) * (j - center_y);
				if (x2 + y2 > radius2)
					result[i][j] = original_mat[i][j] * 0;
				else
					result[i][j] = original_mat[i][j];
			}
		}
		
		return result;

	}

	
	vector<vector<Complex> > IHPF(vector<vector<Complex> >& original_mat, int radius, double a = 0, double b = 1)
	{
		int cols = original_mat[0].size(), rows = original_mat.size();
		int center_x = (cols - 1) / 2, center_y = (rows - 1) / 2;
		int radius2 = radius * radius;
		vector<vector<Complex> > result(original_mat.size(), vector<Complex>(original_mat[0].size()));

		for (int i = 0; i < rows; i++) {
			int x2 = (i - center_x) * (i - center_x);
			for (int j = 0; j < cols; j++) {
				int y2 = (j - center_y) * (j - center_y);
				if (x2 + y2 < radius2)
					result[i][j] = original_mat[i][j] * a;
				else 
					result[i][j] = original_mat[i][j] * (a + b);
			}
		}

		return result;
	}

	vector<vector<Complex> > BLPF(vector<vector<Complex> >& original_mat, int radius, int level = 2)
	{
		int cols = original_mat[0].size(), rows = original_mat.size();
		int center_x = (cols - 1) / 2, center_y = (rows - 1) / 2;
		int radius2 = radius * radius;
		vector<vector<Complex> > result(original_mat.size(), vector<Complex>(original_mat[0].size()));


		for (int i = 0; i < rows; i++) {
			double x2 = (i - center_x) * (i - center_x);
			for (int j = 0; j < cols; j++) {
				double y2 = (j - center_y) * (j - center_y);
				result[i][j] = original_mat[i][j] * (1 / (1 + pow((x2 + y2) / radius2, level)));
			}
		}

		return result;
	}

	vector<vector<Complex> > BHPF(vector<vector<Complex> >& original_mat, int radius, double a = 0, double b = 1, int level = 2)
	{
		int cols = original_mat[0].size(), rows = original_mat.size();
		int center_x = (cols - 1) / 2, center_y = (rows - 1) / 2;
		int radius2 = radius * radius;
		vector<vector<Complex> > result(original_mat.size(), vector<Complex>(original_mat[0].size()));


		for (int i = 0; i < rows; i++) {
			double x2 = (i - center_x) * (i - center_x);
			for (int j = 0; j < cols; j++) {
				double y2 = (j - center_y) * (j - center_y);
				result[i][j] = original_mat[i][j] * ((1 / (1 + pow(radius2 / (x2 + y2), level))) * b + a);
			}
		}

		return result;
	}

	vector<vector<Complex> > GLPF(vector<vector<Complex> >& original_mat, int radius)
	{
		int cols = original_mat[0].size(), rows = original_mat.size();
		int center_x = (cols - 1) / 2, center_y = (rows - 1) / 2;
		double radius2 = 2.0 * radius * radius;
		vector<vector<Complex> > result(original_mat.size(), vector<Complex>(original_mat[0].size()));


		for (int i = 0; i < rows; i++) {
			double x2 = (i - center_x) * (i - center_x);
			for (int j = 0; j < cols; j++) {
				double y2 = (j - center_y) * (j - center_y);
				result[i][j] = original_mat[i][j] * exp(-(x2 + y2) / radius2);
			}
		}

		return result;
	}

	vector<vector<Complex> > GHPF(vector<vector<Complex> >& original_mat, int radius, double a = 0, double b = 1)
	{
		int cols = original_mat[0].size(), rows = original_mat.size();
		int center_x = (cols - 1) / 2, center_y = (rows - 1) / 2;
		double radius2 = 2.0 * radius * radius;
		vector<vector<Complex> > result(original_mat.size(), vector<Complex>(original_mat[0].size()));

		for (int i = 0; i < rows; i++) {
			double x2 = (i - center_x) * (i - center_x);
			for (int j = 0; j < cols; j++) {
				double y2 = (j - center_y) * (j - center_y);
				result[i][j] = original_mat[i][j] * ((1 - exp(-(x2 + y2) / radius2)) * b + a);
			}
		}

		return result;
	}
}