/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face detection, the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"

using namespace std;

int main(int argc, char** argv) {
  if (argc < 3) {
      cout << "Usage: " << argv[0]
          << " image_path model_path"
          << endl;
      return -1;
  }

  const char* img_path = argv[1];				// 待检测图片的路径
  seeta::FaceDetection detector(argv[2]);		// 创建一个检测器

  detector.SetMinFaceSize(40);				// 设置最小检测人脸
  detector.SetScoreThresh(2.f);				// 设置阈值
  detector.SetImagePyramidScaleFactor(0.8f);	// 设置图像金字塔相邻尺度的因子
  detector.SetWindowStep(4, 4);				// 设置滑动窗口在水平、垂直方向上的步距

	// imread()从文件中加载图像并返回该图像。若图像不能被读取（由于文件丢失、无权限、非法格式等原因），返回空矩阵（Mat中data项为nullptr）。
	// 1）图片文件格式支持
	// 该函数根据文件内容确定文件类型，而不是根据扩展名。
	// 在Windows/Mac平台下，OpenCV默认激活了libjpeg, libpng, libtiff, libjasper库，故总是可以读取JPEGs/PNGs/TIFFs格式文件。
	// 支持以下图片格式（前3个always supported）：
	// Windows bitmaps - *.bmp, *.dib：8位，单通道、3通道或4通道输入。
	// Portable image format - *.pbm, *.pgm, *.ppm：NetPBM，8位，单通道（PGM）或3通道（PPM）。
	// PNG - *.png：8位或16位，单通道、3通道或4通道输入。
	// Sun rasters - *.sr, *.ras
	// JPEG files - *.jpeg, *.jpg, *.jpe：基线JPEG；8位，单通道或3通道输入。
	// JPEG 2000 files - *.jp2：8位或16位，单通道或3通道输入。
	// WebP - *.webp
	// TIFF files - *.tiff, *.tif：8位或16位，单通道、3通道或4通道输入。
	// 2）enum cv::ImreadModes 枚举类型说明
	// IMREAD_UNCHANGED：不进行转换，如保存为16位的图片，读出来仍是16位图片。
	// IMREAD_GRAYSCAL：转换为灰度图，如保存为16位图片，读出来是8位图，类型是CV_8UC1。
	// IMREAD_COLOR：转换为三通道图像。
	// IMREAD_ANYDEPTH：若图像深度为16/32位，读出来的是16/32位；其他的转换为8位。
	// IMREAD_ANYCOLOR：读取的通道由具体文件决定，最高3通道。
	// IMREAD_LOAD_GDAL：使用GDAL驱动读取文件，GDAL（Geospatial Data Abstraction Library）在X/MIT许可协议下的开源栅格空间数据转换库。
	//                   它利用抽象数据模型来表达所支持的各种文件格式。它还有一系列命令行工具来进行数据转换和处理。
  cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
  cv::Mat img_gray;			// 灰度图

  if (img.channels() != 1)		// 若通道数不是1，则需要通过 cv::cvtColor() 来转换
	  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  else                            // 若通道数为1，则读进来的图本身就是灰度图，直接赋值即可 
	  img_gray = img;

  seeta::ImageData img_data;
  img_data.data = img_gray.data;		// 灰度图的数据赋值
  img_data.width = img_gray.cols;		// 灰度图的列数，即为图像的宽度
  img_data.height = img_gray.rows;	// 灰度图的行数，即为图像的高度
  img_data.num_channels = 1;			// 灰度图的通道数固定为1

  cv::imshow("灰度图", img_gray);

  long t0 = cv::getTickCount();

  // 检测人脸
  // 检测到的人脸数据已经存储在 FaceInfo.bbox 中
  std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
  long t1 = cv::getTickCount();
  double secs = (t1 - t0)/cv::getTickFrequency();

  cout << "Detections takes " << secs << " seconds " << endl;
#ifdef USE_OPENMP
  cout << "OpenMP is used." << endl;
#else
  cout << "OpenMP is not used. " << endl;
#endif

#ifdef USE_SSE
  cout << "SSE is used." << endl;
#else
  cout << "SSE is not used." << endl;
#endif

  cout << "Image size (wxh): " << img_data.width << "x" 
      << img_data.height << endl;

  // 输出检测到的人脸数
  cout << "Face size:" << faces.size() << endl;

  // 给检测到的人脸画框
  cv::Rect face_rect;
  // size_t 静态类型转换为 int32_t 类型
  int32_t num_face = static_cast<int32_t>(faces.size());

  for (int32_t i = 0; i < num_face; i++) {
	// 获取绘制矩形所需要的参数：原点（左上角）坐标、高度、宽度。
    face_rect.x = faces[i].bbox.x;
    face_rect.y = faces[i].bbox.y;
    face_rect.width = faces[i].bbox.width;
    face_rect.height = faces[i].bbox.height;

	// 绘制矩形
    cv::rectangle(img, face_rect, CV_RGB(0, 255, 255), 1, 8, 0);
  }

  cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);

  // imshow显示的图像是由整数（CV_16U表示16为无符号整数，CV_32S表示32为有符号整数）构成，
  // 图像每个像素的值会被除以256，以便能够在256级灰度中显示。在显示由浮点数构成的图像时，
  // 值的范围会被假设为0.0(黑色)到1.0（白色）之间，大于1.0显示白色，小于0.0显示黑色。
  cv::imshow("Test", img);
  cv::waitKey(0);
  cv::destroyAllWindows();
}
