#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <ctime>
#include <chrono>

#include <cmath>

#include "pthread.h"

#define THREADS_NUM 4
#define REPEAT_NUM 500

typedef struct {
	int start_index;
	int end_index;
	unsigned char* first_frame;
	unsigned char* second_frame;
	unsigned char* output_frame;
} input_package;

typedef struct {
	int start_row;
	int end_row;
	int pic2_col_size;
	int pic2_row_size;
	int pic1_col_size;
	int pic1_row_size;
	unsigned char* first_frame;
	unsigned char* second_frame;
	unsigned char* output_frame;
} input_package_Q4;


void* Q4_thread_task(void* arg) {
	input_package_Q4* inp = (input_package_Q4*)arg;
	for (int i = inp->start_row; i < inp->end_row; i++) {
		for (int j = 0; j < inp->pic1_col_size; j++) {
			int index1 = i*(inp->pic1_col_size) + j;
			if(i > inp->pic2_row_size || j > inp->pic2_col_size)
				*(inp->output_frame + index1) = *(inp->first_frame + index1);
			else {
				int index2 = i*(inp->pic2_col_size) + j;
				long result = *(inp->first_frame + index1) + (*(inp->second_frame + index2) / 4);
				if (result < 255)
					*(inp->output_frame + index1) = result;
				else
					*(inp->output_frame + index1) = 255;
			}
		}
	}
	pthread_exit(NULL);
	return NULL;
}

void* Q3_thread_task(void* arg) {
	input_package* inp = (input_package*)arg;
	for (int i = inp->start_index; i < inp->end_index; i++) {
		*(inp->output_frame + i) = abs(*(inp->first_frame + i) - *(inp->second_frame + i));
	}
	pthread_exit(NULL);
	return NULL;
}



double Q3_serial() {
	double ser_duration = 0.0;

	cv::Mat first_frame = cv::imread("Q3_frame1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_frame = cv::imread("Q3_frame2.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_frame_ser(first_frame.rows, first_frame.cols, CV_8U);
	cv::Mat out_frame_par(first_frame.rows, first_frame.cols, CV_8U);
	int NROWS = first_frame.rows;
	int NCOLS = first_frame.cols;

	unsigned char *frame_one, *frame_two, *output_frame;
	frame_one = (unsigned char *)first_frame.data;
	frame_two = (unsigned char *)second_frame.data;
	output_frame = (unsigned char *)out_frame_ser.data;

	double sum_duration = 0.0;
	for (int i = 0; i < REPEAT_NUM; i++) {
		auto start_ser = std::chrono::high_resolution_clock::now();
		for (int row = 0; row < NROWS; row++) {
			for (int col = 0; col < NCOLS; col++) {
				*(output_frame + row*NCOLS + col) = abs(*(frame_one + row*NCOLS + col) - *(frame_two + row*NCOLS + col));
			}
		}
		auto end_ser = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_s = end_ser - start_ser;
		sum_duration += duration_s.count();
	}
	cv::imshow("Q3 image serial", out_frame_ser);

	ser_duration = sum_duration / REPEAT_NUM;
	printf("Average serial execution time of Q3 for %d execution: %f\n", REPEAT_NUM, ser_duration);

	cv::waitKey(0);
	return ser_duration;
}

double Q3_parallel() {
	double par_duration = 0.0;

	cv::Mat first_frame = cv::imread("Q3_frame1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_frame = cv::imread("Q3_frame2.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_frame_par(first_frame.rows, first_frame.cols, CV_8U);
	long NROWS = first_frame.rows;
	long NCOLS = first_frame.cols;

	unsigned char *frame_one, *frame_two, *output_frame;
	frame_one = (unsigned char *)first_frame.data;
	frame_two = (unsigned char *)second_frame.data;
	output_frame = (unsigned char *)out_frame_par.data;

	double all_pixel_num = NROWS * NCOLS;
	long package_size = ceil(all_pixel_num / THREADS_NUM);

	double sum_duration = 0.0;
	for (int i = 0; i < REPEAT_NUM; i++) {
		auto start_par = std::chrono::high_resolution_clock::now();
		pthread_t thread[THREADS_NUM];
		input_package input_args[THREADS_NUM];
		int thread_index = 0;
		for (long i = 0; i < all_pixel_num; i += package_size) {
			input_args[thread_index].first_frame = frame_one;
			input_args[thread_index].second_frame = frame_two;
			input_args[thread_index].output_frame = output_frame;
			input_args[thread_index].start_index = i;
			if (i + package_size < all_pixel_num)
				input_args[thread_index].end_index = i + package_size;
			else
				input_args[thread_index].end_index = all_pixel_num;
			pthread_create(&thread[thread_index], NULL, Q3_thread_task, &input_args[thread_index]);
			thread_index++;
		}
		for (int i = 0; i < THREADS_NUM; i++) {
			pthread_join(thread[i], NULL);
		}
		auto end_par = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_s = end_par - start_par;
		sum_duration += duration_s.count();
	}

	cv::imshow("Q3 image parallel", out_frame_par);

	par_duration = sum_duration / REPEAT_NUM;
	printf("Average Parallel execution time of Q3 for %d execution: %f\n", REPEAT_NUM, par_duration);

	cv::waitKey(0);
	return par_duration;
}


double Q4_serial() {
	cv::Mat first_pic = cv::imread("CA#02__Image__01.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_pic = cv::imread("CA#02__Image__02.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_frame_ser(first_pic.rows, first_pic.cols, CV_8U);
	double ser_duration = 0.0;

	int NROWS_pic1 = first_pic.rows;
	int NROWS_pic2 = second_pic.rows;
	int NCOLS_pic2 = second_pic.cols;
	int NCOLS_pic1 = first_pic.cols;

	unsigned char *frame_one, *frame_two, *output_frame;
	frame_one = (unsigned char *)first_pic.data;
	frame_two = (unsigned char *)second_pic.data;
	output_frame = (unsigned char *)out_frame_ser.data;

	double sum_duration = 0.0;
	for (int r = 0; r < REPEAT_NUM; r++) {
		auto start_par = std::chrono::high_resolution_clock::now();
		for (int row = 0; row < NROWS_pic1; row++) {
			for (int col = 0; col < NCOLS_pic1; col++) {
				if(row > NROWS_pic2 || col > NCOLS_pic2)
					*(output_frame + row*NCOLS_pic1 + col) = *(frame_one + row*NCOLS_pic1 + col);
				else {
					int sum = *(frame_one + row*NCOLS_pic1 + col) + (*(frame_two + row*NCOLS_pic2 + col) / 4);
					if (sum > 255)
						*(output_frame + row*NCOLS_pic1 + col) = 255;
					else
						*(output_frame + row*NCOLS_pic1 + col) = sum;
				}
			}
		}
		auto end_par = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_s = end_par - start_par;
		sum_duration += duration_s.count();
	}

	cv::imshow("Q4 image serial", out_frame_ser);
	ser_duration = sum_duration / REPEAT_NUM;
	printf("Average serial execution time of Q4 for %d execution: %f\n", REPEAT_NUM, ser_duration);

	cv::waitKey(0);
	return ser_duration;
}

double Q4_parallel() {
	cv::Mat first_pic = cv::imread("CA#02__Image__01.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_pic = cv::imread("CA#02__Image__02.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_frame_par(first_pic.rows, first_pic.cols, CV_8U);
	double par_duration = 0.0;

	int NROWS_pic1 = first_pic.rows;
	int NROWS_pic2 = second_pic.rows;
	int NCOLS_pic2 = second_pic.cols;
	int NCOLS_pic1 = first_pic.cols;
	int package_rows_size = 0.0;
	package_rows_size = ceil(NROWS_pic1 / THREADS_NUM);

	unsigned char *frame_one, *frame_two, *output_frame;
	frame_one = (unsigned char *)first_pic.data;
	frame_two = (unsigned char *)second_pic.data;
	output_frame = (unsigned char *)out_frame_par.data;

	double sum_duration = 0.0;

	for (int r = 0; r < REPEAT_NUM; r++) {
		auto start_par = std::chrono::high_resolution_clock::now();
		pthread_t thread[THREADS_NUM];
		input_package_Q4 input_args[THREADS_NUM];
		int thread_index = 0;
		for (long i = 0; i < NROWS_pic1; i += package_rows_size) {
			input_args[thread_index].pic1_col_size = NCOLS_pic1;
			input_args[thread_index].pic1_row_size = NROWS_pic1;
			input_args[thread_index].pic2_col_size = NCOLS_pic2;
			input_args[thread_index].pic2_row_size = NROWS_pic2;
			input_args[thread_index].first_frame = frame_one;
			input_args[thread_index].second_frame = frame_two;
			input_args[thread_index].output_frame = output_frame;
			input_args[thread_index].start_row = i;
			if (i + package_rows_size < NROWS_pic1)
				input_args[thread_index].end_row = i + package_rows_size;
			else
				input_args[thread_index].end_row = NROWS_pic1;
			pthread_create(&thread[thread_index], NULL, Q4_thread_task, &input_args[thread_index]);
			thread_index++;
		}
		for (int i = 0; i < THREADS_NUM; i++) {
			pthread_join(thread[i], NULL);
		}
		auto end_par = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_s = end_par - start_par;
		sum_duration += duration_s.count();
	}

	cv::imshow("Q4 image parallel", out_frame_par);
	par_duration = sum_duration / REPEAT_NUM;
	printf("Average Parallel execution time of Q4 for %d execution: %f\n", REPEAT_NUM, par_duration);

	cv::waitKey(0);
	return par_duration;
}



int main() {
	double parallel_dur_Q3 = Q3_parallel();
	double serial_dur_Q3 = Q3_serial();
	printf("speed up Q3: %f\n", serial_dur_Q3 / parallel_dur_Q3);
	double serial_dur_Q4 = Q4_serial();
	double parallel_dur_Q4 = Q4_parallel();
	printf("speed up Q4: %f\n", serial_dur_Q4 / parallel_dur_Q4);

	return 0;
}