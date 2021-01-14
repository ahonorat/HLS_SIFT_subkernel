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

    xf::cv::subtract<XF_CONVERT_POLICY_SATURATE, TYPE, HEIGHT, WIDTH, NPC1, MAXDELAY*NBBLURS>(imgInputDup2, imgBlurred3, imgOutput);

    // Convert output xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel

} // End of extern C
