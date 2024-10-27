#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <time.h>
#include <iostream>
#include <ctime>
#include <chrono>

#include "pthread.h"




void Q4() {
	cv::Mat first_pic_par = cv::imread("CA#02__Image__01.png", cv::IMREAD_GRAYSCALE);
	cv::Mat first_pic_ser = cv::imread("CA#02__Image__01.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_pic = cv::imread("CA#02__Image__02.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_pic_ser(first_pic_ser.rows, first_pic_ser.cols, CV_8U);
	cv::Mat out_pic_par(first_pic_ser.rows, first_pic_ser.cols, CV_8U);

	int NROWS_pic1 = first_pic_ser.rows;
	int NROWS_pic2 = second_pic.rows;
	int NCOLS_pic2 = second_pic.cols;
	int NCOLS_pic1 = first_pic_ser.cols;

	double ser_duration = 0.0, par_duration = 0.0;

	//serial implementation

	//Ipp64u start, end;
	//float serial_duration;
	//start = ippGetCpuClocks();

	//time_t start_ser, end_ser;


	unsigned char *frame_one, *frame_two;
	frame_one = (unsigned char *)first_pic_ser.data;
	frame_two = (unsigned char *)second_pic.data;

	auto start_ser = std::chrono::high_resolution_clock::now();

	for (int row = 0; row < NROWS_pic2; row++) {
		for (int col = 0; col < NCOLS_pic2; col++) {
			int sum = *(frame_one + row*NCOLS_pic1 + col) + ((*(frame_two + row*NCOLS_pic2 + col) / 4));
			if (sum > 255)
				*(frame_one + row*NCOLS_pic1 + col) = 255;
			else
				*(frame_one + row*NCOLS_pic1 + col) = sum;
		}
	}

	auto end_ser = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_s = end_ser - start_ser;
	ser_duration = duration_s.count();
	printf("Serial execution time of Q4 in seconds: %f\n", ser_duration);
	//end = ippGetCpuClocks();
	//serial_duration = (Ipp32s)(end - start);
	cv::imshow("Q4 image serial", first_pic_ser);
	cv::waitKey(0);


	//parallel implementation
	//time_t start_par, end_par;
	unsigned char *frame_one_par, *frame_two_par;
	frame_one_par = (unsigned char *)first_pic_ser.data;
	frame_two_par = (unsigned char *)second_pic.data;
	int row = 0;

	//start_par = clock();
	auto start_par = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(2) shared(frame_one_par, frame_two_par) private(row)
	{
#pragma omp for
		for (row = 0; row < NROWS_pic2; row++) {
			for (int col = 0; col < NCOLS_pic2; col++) {
				int sum = *(frame_one_par + row*NCOLS_pic1 + col) + ((*(frame_two_par + row*NCOLS_pic2 + col) / 4));
				if (sum > 255)
					*(frame_one_par + row*NCOLS_pic1 + col) = 255;
				else
					*(frame_one_par + row*NCOLS_pic1 + col) = sum;
			}
		}
	}
	//end_par = clock();
	//par_duration = (double)(end_par - start_par) / CLOCKS_PER_SEC;
	auto end_par = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_p = end_par - start_par;
	par_duration = duration_p.count();

	printf("Parallel execution time of Q4 in seconds: %f\n", par_duration);
	//end = ippGetCpuClocks();
	//serial_duration = (Ipp32s)(end - start);
	cv::imshow("Q4 image parallel", first_pic_ser);
	cv::waitKey(0);

	double efficiency = 0.0;
	efficiency = ser_duration / par_duration;
	printf("OMP efficiency Q4: %f\n", efficiency);
}

void Q3() {
#pragma omp parallel num_threads(3)
	printf("hello world!\n");

	cv::Mat first_frame = cv::imread("Q3_frame1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_frame = cv::imread("Q3_frame2.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_frame_ser(first_frame.rows, first_frame.cols, CV_8U);
	cv::Mat out_frame_par(first_frame.rows, first_frame.cols, CV_8U);
	int NROWS = first_frame.rows;
	int NCOLS = first_frame.cols;

	double par_duration = 0.0, ser_duration = 0.0, efficiency = 0.0;

	//serial implementation

	//Ipp64u start, end;
	//float serial_duration;
	//start = ippGetCpuClocks();
	time_t start_ser, end_ser;
	unsigned char *frame_one, *frame_two, *output_frame;
	frame_one = (unsigned char *)first_frame.data;
	frame_two = (unsigned char *)second_frame.data;
	output_frame = (unsigned char *)out_frame_ser.data;

	//struct timeval start_ser;
	//struct timeval end_ser;

	//gettimeofday(&start_ser, NULL);

	start_ser = clock();

	for (int row = 0; row < NROWS; row++) {
		for (int col = 0; col < NCOLS; col++) {
			*(output_frame + row*NCOLS + col) = abs(*(frame_one + row*NCOLS + col) - *(frame_two + row*NCOLS + col));
		}
	}

	end_ser = clock();

	ser_duration = (double)(end_ser - start_ser) / CLOCKS_PER_SEC;
	printf("Serial execution time of Q3 in seconds: %f\n", ser_duration);
	//gettimeofday(&end_ser, NULL);
	//long seconds_ser = 0;
	//seconds_ser = end_ser.tv_sec - start_ser.tv_sec;
	//printf("Serial execution time of Q3 in seconds : %ld\n", seconds_ser);

	//end = ippGetCpuClocks();
	//serial_duration = (Ipp32s)(end - start);
	cv::imshow("Q3 image serial", out_frame_ser);
	cv::waitKey(0);

	//parallel implementation

	//Ipp64u start, end;
	//float serial_duration;
	//start = ippGetCpuClocks();

	time_t start_par, end_par;
	unsigned char *frame_one_par, *frame_two_par, *output_frame_par;
	frame_one_par = (unsigned char *)first_frame.data;
	frame_two_par = (unsigned char *)second_frame.data;
	output_frame_par = (unsigned char *)out_frame_ser.data;

	start_par = clock();
	int row = 0;
	#pragma omp parallel num_threads(2) shared(output_frame_par, frame_one_par, frame_two_par) private(row)
	{
	#pragma omp parallel for
		for (row = 0; row < NROWS; row++) {
			for (int col = 0; col < NCOLS; col++) {
				*(output_frame_par + row*NCOLS + col) = abs(*(frame_one_par + row*NCOLS + col) - *(frame_two_par + row*NCOLS + col));
			}
		}
	}

	end_par = clock();
	par_duration = (double)(end_par - start_par) / CLOCKS_PER_SEC;
	printf("Parallel execution time of Q3 in seconds: %f\n", par_duration);
	//end = ippGetCpuClocks();
	//serial_duration = (Ipp32s)(end - start);
	cv::imshow("Q3 image parallel", out_frame_ser);
	cv::waitKey(0);

	efficiency = ser_duration / par_duration;
	printf("OMP efficiency: %f\n", efficiency);

}


void Q4_1000_execution() {
	

	//serial implementation

	//Ipp64u start, end;
	//float serial_duration;
	//start = ippGetCpuClocks();

	//time_t start_ser, end_ser;

	cv::Mat first_pic_par = cv::imread("CA#02__Image__01.png", cv::IMREAD_GRAYSCALE);
	cv::Mat first_pic_ser = cv::imread("CA#02__Image__01.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_pic = cv::imread("CA#02__Image__02.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_pic_ser(first_pic_ser.rows, first_pic_ser.cols, CV_8U);
	cv::Mat out_pic_par(first_pic_ser.rows, first_pic_ser.cols, CV_8U);

	int NROWS_pic1 = first_pic_ser.rows;
	int NROWS_pic2 = second_pic.rows;
	int NCOLS_pic2 = second_pic.cols;
	int NCOLS_pic1 = first_pic_ser.cols;

	double sum = 0.0;
	for (int iter = 0; iter < 1000; iter++) {
		

		double ser_duration = 0.0, par_duration = 0.0;


		unsigned char *frame_one, *frame_two;
		frame_one = (unsigned char *)first_pic_ser.data;
		frame_two = (unsigned char *)second_pic.data;


		auto start_ser = std::chrono::high_resolution_clock::now();

		for (int row = 0; row < NROWS_pic2; row++) {
			for (int col = 0; col < NCOLS_pic2; col++) {
				int sum = *(frame_one + row*NCOLS_pic1 + col) + ((*(frame_two + row*NCOLS_pic2 + col) / 4));
				if (sum > 255)
					*(frame_one + row*NCOLS_pic1 + col) = 255;
				else
					*(frame_one + row*NCOLS_pic1 + col) = sum;
			}
		}

		auto end_ser = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_s = end_ser - start_ser;
		ser_duration = duration_s.count();
		//printf("Serial execution time of Q4 in seconds: %f\n", ser_duration);
		//end = ippGetCpuClocks();
		//serial_duration = (Ipp32s)(end - start);


		//parallel implementation
		//time_t start_par, end_par;
		unsigned char *frame_one_par, *frame_two_par;
		frame_one_par = (unsigned char *)first_pic_ser.data;
		frame_two_par = (unsigned char *)second_pic.data;
		int row = 0;

		//start_par = clock();
		auto start_par = std::chrono::high_resolution_clock::now();

		#pragma omp parallel num_threads(2) shared(frame_one_par, frame_two_par) private(row)
		{
		#pragma omp for
			for (row = 0; row < NROWS_pic2; row++) {
				for (int col = 0; col < NCOLS_pic2; col++) {
					int sum = *(frame_one_par + row*NCOLS_pic1 + col) + ((*(frame_two_par + row*NCOLS_pic2 + col) / 4));
					if (sum > 255)
						*(frame_one_par + row*NCOLS_pic1 + col) = 255;
					else
						*(frame_one_par + row*NCOLS_pic1 + col) = sum;
				}
			}
		}
		//end_par = clock();
		//par_duration = (double)(end_par - start_par) / CLOCKS_PER_SEC;
		auto end_par = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_p = end_par - start_par;
		par_duration = duration_p.count();

		//printf("Parallel execution time of Q4 in seconds: %f\n", par_duration);
		//end = ippGetCpuClocks();
		//serial_duration = (Ipp32s)(end - start);
		
		double efficiency = 0.0;
		efficiency = ser_duration / par_duration;
		sum += efficiency;

		//cv::imshow("Q4 image serial", first_pic_ser);
		//cv::imshow("Q4 image parallel", first_pic_ser);

	}
	printf("average OMP efficiency Q4 for 1000 execution: %f\n", (sum/1000));
	cv::waitKey(0);
}

void Q3_1000_execution() {

	cv::Mat first_frame = cv::imread("Q3_frame1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_frame = cv::imread("Q3_frame2.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_frame_ser(first_frame.rows, first_frame.cols, CV_8U);
	cv::Mat out_frame_par(first_frame.rows, first_frame.cols, CV_8U);
	int NROWS = first_frame.rows;
	int NCOLS = first_frame.cols;

	double sum = 0.0;
	for (int index = 0; index < 1000; index++) {
		double par_duration = 0.0, ser_duration = 0.0, efficiency = 0.0;

		//serial implementation

		
		unsigned char *frame_one, *frame_two, *output_frame;
		frame_one = (unsigned char *)first_frame.data;
		frame_two = (unsigned char *)second_frame.data;
		output_frame = (unsigned char *)out_frame_ser.data;

		//struct timeval start_ser;
		//struct timeval end_ser;

		//gettimeofday(&start_ser, NULL);

		auto start_ser = std::chrono::high_resolution_clock::now();

		for (int row = 0; row < NROWS; row++) {
			for (int col = 0; col < NCOLS; col++) {
				*(output_frame + row*NCOLS + col) = abs(*(frame_one + row*NCOLS + col) - *(frame_two + row*NCOLS + col));
			}
		}

		auto end_ser = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_s = end_ser - start_ser;
		ser_duration = duration_s.count();
		

		//parallel implementation

		unsigned char *frame_one_par, *frame_two_par, *output_frame_par;
		frame_one_par = (unsigned char *)first_frame.data;
		frame_two_par = (unsigned char *)second_frame.data;
		output_frame_par = (unsigned char *)out_frame_ser.data;

		auto start_par = std::chrono::high_resolution_clock::now();
		int row = 0;
		#pragma omp parallel num_threads(2)
		{
			#pragma omp parallel for
			for (row = 0; row < NROWS; row++) {
				for (int col = 0; col < NCOLS; col++) {
					*(output_frame_par + row*NCOLS + col) = abs(*(frame_one_par + row*NCOLS + col) - *(frame_two_par + row*NCOLS + col));
				}
			}
		}

		auto end_par = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_p = end_par - start_par;
		par_duration = duration_p.count();

		efficiency = ser_duration / par_duration;
		sum += efficiency;
	}
	
	printf("Average OMP efficiency of Q3 for 1000 execution: %f\n", (sum/1000));
}

int main() {
	Q3();
	Q4();
	Q3_1000_execution();
	Q4_1000_execution();

	return 0;
}