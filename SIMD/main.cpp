#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <intrin.h>
//#include "ipp.h"



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

	//serial implementation

	//Ipp64u start, end;
	//float serial_duration;
	//start = ippGetCpuClocks();
	unsigned char *frame_one, *frame_two;
	frame_one = (unsigned char *)first_pic_ser.data;
	frame_two = (unsigned char *)second_pic.data;
	for (int row = 0; row < NROWS_pic2; row++) {
		for (int col = 0; col < NCOLS_pic2; col++) {
			int sum = *(frame_one + row*NCOLS_pic1 + col) + (*(frame_two + row*NCOLS_pic2 + col)/4);
			if (sum > 255)
				*(frame_one + row*NCOLS_pic1 + col) = 255;
			else
				*(frame_one + row*NCOLS_pic1 + col) = sum;
		}
	}
	//end = ippGetCpuClocks();
	//serial_duration = (Ipp32s)(end - start);
	cv::imshow("Q4 image serial", first_pic_ser);
	cv::waitKey(0);


	//parallel implementation

	//float parallel_duration;
	//start = ippGetCpuClocks();
	__m128i *frame_1, *frame_2;
	frame_1 = (__m128i *) first_pic_par.data;
	frame_2 = (__m128i *) second_pic.data;
	__m128i result, const_num2, and_index, pic1_var, pic2_var;
	const_num2 = _mm_set_epi32(0, 0, 0, 2);
	and_index = _mm_set1_epi8(63);
	for (int row = 0; row < NROWS_pic2; row++) {
		for (int col = 0; col < NCOLS_pic2 / 16; col++) {
			pic1_var = _mm_loadu_si128(frame_1 + row*NCOLS_pic1 / 16 + col);
			pic2_var = _mm_loadu_si128(frame_2 + row*NCOLS_pic2 / 16 + col);
			result = _mm_sra_epi32(pic2_var, const_num2);
			result = _mm_and_si128(result, and_index);
			result = _mm_adds_epu8(pic1_var, result);
			_mm_storeu_si128(frame_1 + row*NCOLS_pic1 / 16 + col, result);
		}
	}
	//end = ippGetCpuClocks();
	//parallel_duration = (Ipp32s)(end - start);
	//printf("**********Q3**********\nserial execution time: %f\nparallel execution time: %f\nsppedup: %f",
		   //serial_duration, parallel_duration, serial_duration/parallel_duration);  
	cv::imshow("Q4 image parallel", first_pic_par);
	cv::waitKey(0);
}

void Q3() {
	cv::Mat first_frame = cv::imread("Q3_frame1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat second_frame = cv::imread("Q3_frame2.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out_frame_ser(first_frame.rows, first_frame.cols, CV_8U);
	cv::Mat out_frame_par(first_frame.rows, first_frame.cols, CV_8U);
	int NROWS = first_frame.rows;
	int NCOLS = first_frame.cols;

	//serial implementation

	//Ipp64u start, end;
	//float serial_duration;
	//start = ippGetCpuClocks();
	unsigned char *frame_one, *frame_two, *output_frame;
	frame_one = (unsigned char *)first_frame.data;
	frame_two = (unsigned char *)second_frame.data;
	output_frame = (unsigned char *)out_frame_ser.data;
	for (int row = 0; row < NROWS; row++) {
		for (int col = 0; col < NCOLS; col++) {
			*(output_frame + row*NCOLS + col) = abs(*(frame_one + row*NCOLS + col) - *(frame_two + row*NCOLS + col));
		}
	}
	//end = ippGetCpuClocks();
	//serial_duration = (Ipp32s)(end - start);
	cv::imshow("Q3 image serial", out_frame_ser);
	cv::waitKey(0);

	//parallel implementation

	//float parallel_duration;
	//start = ippGetCpuClocks();
	__m128i *frame_1, *frame_2, *result_frame;
	frame_1 = (__m128i *) first_frame.data;
	frame_2 = (__m128i *) second_frame.data;
	result_frame = (__m128i *) out_frame_par.data;
	__m128i load_value1, load_value2, temp_result, result;
	for (int row = 0; row < NROWS; row++) {
		for (int col = 0; col < NCOLS / 16; col++) {
			load_value1 = _mm_loadu_si128(frame_1 + row*NCOLS / 16 + col);
			load_value2 = _mm_loadu_si128(frame_2 + row*NCOLS / 16 + col);
			temp_result = _mm_subs_epi8(load_value1, load_value2);
			result = _mm_abs_epi8(temp_result);
			_mm_storeu_si128(result_frame + row*NCOLS / 16 + col, result);
		}
	}
	//end = ippGetCpuClocks();
	//parallel_duration = (Ipp32s)(end - start);
	cv::imshow("Q3 image parallel", out_frame_par);
	cv::waitKey(0);

}

int main() {

	Q3();
	Q4();

	return 0;
}