/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xf_gaussian_diff_config.h"



template<int T, int ROWS, int COLS, int NPC>
void GradRot(xf::cv::Mat<T, ROWS, COLS, NPC> &ImgIn,
		  xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> &ImgGrad,
		  xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> &ImgRot) {

#pragma HLS DATAFLOW
	int rows = ImgIn.rows;
	int cols = ImgIn.cols;
    xf::cv::Mat<T, ROWS, COLS, NPC> imgInputDup_0(rows, cols);
    xf::cv::Mat<T, ROWS, COLS, NPC> imgInputDup_1(rows, cols);
//    xf::cv::Mat<T, ROWS, COLS, NPC> imgFilteredRows(rows, cols);
//    xf::cv::Mat<T, ROWS, COLS, NPC> imgFilteredCols(rows, cols);
    xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> imgFilteredRows16SC1(rows, cols);
    xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> imgFilteredCols16SC1(rows, cols);
    xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> imgFilteredRowsDup_0(rows, cols);
    xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> imgFilteredColsDup_0(rows, cols);
    xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> imgFilteredRowsDup_1(rows, cols);
    xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> imgFilteredColsDup_1(rows, cols);

    xf::cv::duplicateMat<T, ROWS, COLS, NPC>(ImgIn, imgInputDup_0, imgInputDup_1);
    short int filterDiffRows[9] = {0, 0, 0, 1, 0, -1, 0, 0, 0};
    short int filterDiffCols[9] = {0, 1, 0, 0, 0, 0, 0, -1, 0};
    xf::cv::filter2D<XF_BORDER_CONSTANT, 3, 3, T, XF_16SC1, ROWS, COLS, NPC>(imgInputDup_0, imgFilteredRows16SC1, filterDiffRows, 0);
    // done by the filter
    //    imgFilteredRows.convertTo<XF_16SC1>(imgFilterRows16SC1, XF_CONVERT_8U_TO_16S);
    xf::cv::duplicateMat<XF_16SC1, ROWS, COLS, NPC>(imgFilteredRows16SC1, imgFilteredRowsDup_0, imgFilteredRowsDup_1);
    xf::cv::filter2D<XF_BORDER_CONSTANT, 3, 3, T, XF_16SC1, ROWS, COLS, NPC>(imgInputDup_1, imgFilteredCols16SC1, filterDiffCols, 0);
    // done by the filter
    //    imgFilteredCols.convertTo<XF_16SC1>(imgFilterCols16SC1, XF_CONVERT_8U_TO_16S);
    xf::cv::duplicateMat<XF_16SC1, ROWS, COLS, NPC>(imgFilteredCols16SC1, imgFilteredColsDup_0, imgFilteredColsDup_1);

    xf::cv::magnitude<XF_L2NORM, XF_16SC1, XF_16SC1, ROWS, COLS, NPC>(imgFilteredColsDup_0, imgFilteredRowsDup_0, ImgGrad);
    xf::cv::phase<XF_RADIANS, XF_16SC1, XF_16SC1, ROWS, COLS, NPC>(imgFilteredColsDup_1, imgFilteredRowsDup_1, ImgRot);

}

template<int T, int ROWS, int COLS, int NPC>
void BlurSub (xf::cv::Mat<T, ROWS, COLS, NPC> &ImgIn,
			  xf::cv::Mat<T, ROWS, COLS, NPC> &ImgBlurred,
			  xf::cv::Mat<T, ROWS, COLS, NPC> &ImgSub,
			  float sigma) {
#pragma HLS DATAFLOW
	int rows = ImgIn.rows;
	int cols = ImgIn.cols;
    xf::cv::Mat<T, ROWS, COLS, NPC> imgInputDup_0_0(rows, cols);
    xf::cv::Mat<T, ROWS, COLS, NPC, MAXDELAY> imgInputDup_0_1(rows, cols);
    xf::cv::Mat<T, ROWS, COLS, NPC> imgBlurredTmp(rows, cols);
    xf::cv::Mat<T, ROWS, COLS, NPC> imgBlurredDup(rows, cols);

    xf::cv::duplicateMat<T, ROWS, COLS, NPC, MAXDELAY>(ImgIn, imgInputDup_0_0, imgInputDup_0_1);
    xf::cv::GaussianBlur<FILTER_WIDTH, XF_BORDER_CONSTANT, T, ROWS, COLS, NPC>(imgInputDup_0_0, imgBlurredTmp, sigma);
    xf::cv::duplicateMat<T, ROWS, COLS, NPC>(imgBlurredTmp, imgBlurredDup, ImgBlurred);
    xf::cv::subtract<XF_CONVERT_POLICY_SATURATE, T, ROWS, COLS, NPC, MAXDELAY>(imgInputDup_0_1, imgBlurredDup, ImgSub);

}


extern "C" {

void gaussiandiference(ap_uint<PTR_WIDTH>* img_in, float sigma, ap_uint<PTR_WIDTH>* img_out, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem1  
    #pragma HLS INTERFACE s_axilite  port=sigma 			          
	#pragma HLS INTERFACE s_axilite  port=rows 			          
	#pragma HLS INTERFACE s_axilite  port=cols 			          
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInputDup(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1, MAXDELAY*NBBLURS> imgInputDup2(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgOutput(rows, cols);

    // could it be iterated over?
	// https://www.codeproject.com/Articles/857354/Compile-Time-Loops-with-Cplusplus-Creating-a-Gener
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgBlurred1(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgSub1(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgBlurred2(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgSub2(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgBlurred3(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgSub3(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgSubGlob(rows, cols);

    xf::cv::Mat<XF_16SC1, HEIGHT, WIDTH, NPC1> imgRotGlob(rows, cols);
    xf::cv::Mat<XF_16SC1, HEIGHT, WIDTH, NPC1> imgGrdGlob(rows, cols);


// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::duplicateMat<TYPE, HEIGHT, WIDTH, NPC1, MAXDELAY*NBBLURS>(imgInput, imgInputDup, imgInputDup2);

    // NBBLURS=3
    BlurSub<TYPE, HEIGHT, WIDTH, NPC1>(imgInputDup, imgBlurred1, imgSub1, sigma);
    BlurSub<TYPE, HEIGHT, WIDTH, NPC1>(imgBlurred1, imgBlurred2, imgSub2, sigma);
    BlurSub<TYPE, HEIGHT, WIDTH, NPC1>(imgBlurred2, imgBlurred3, imgSub3, sigma);

    xf::cv::subtract<XF_CONVERT_POLICY_SATURATE, TYPE, HEIGHT, WIDTH, NPC1, MAXDELAY*NBBLURS>(imgInputDup2, imgBlurred3, imgSubGlob);

    GradRot<TYPE, HEIGHT, WIDTH, NPC1>(imgSubGlob, imgGrdGlob, imgRotGlob);
    // does not compile:
    //imgRotGlob.convertTo<TYPE>(imgOutput, XF_CONVERT_16S_TO_8U);
    // shift: 0 or more if we want to divide
    xf::cv::convertTo<XF_16SC1,XF_8UC1,HEIGHT,WIDTH,NPC1>(imgGrdGlob, imgOutput, XF_CONVERT_16S_TO_8U, 1);

    // should call to synshronize imSubX for detectKeypoints
//    xf::cv::delayMat<MAXDELAY,TYPE,HEIGHT,WIDTH,NPC1>(_src, _dst);

    // Convert output xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel

} // End of extern C
