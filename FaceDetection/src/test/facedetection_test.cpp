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

  const char* img_path = argv[1];				// �����ͼƬ��·��
  seeta::FaceDetection detector(argv[2]);		// ����һ�������

  detector.SetMinFaceSize(40);				// ������С�������
  detector.SetScoreThresh(2.f);				// ������ֵ
  detector.SetImagePyramidScaleFactor(0.8f);	// ����ͼ����������ڳ߶ȵ�����
  detector.SetWindowStep(4, 4);				// ���û���������ˮƽ����ֱ�����ϵĲ���

	// imread()���ļ��м���ͼ�񲢷��ظ�ͼ����ͼ���ܱ���ȡ�������ļ���ʧ����Ȩ�ޡ��Ƿ���ʽ��ԭ�򣩣����ؿվ���Mat��data��Ϊnullptr����
	// 1��ͼƬ�ļ���ʽ֧��
	// �ú��������ļ�����ȷ���ļ����ͣ������Ǹ�����չ����
	// ��Windows/Macƽ̨�£�OpenCVĬ�ϼ�����libjpeg, libpng, libtiff, libjasper�⣬�����ǿ��Զ�ȡJPEGs/PNGs/TIFFs��ʽ�ļ���
	// ֧������ͼƬ��ʽ��ǰ3��always supported����
	// Windows bitmaps - *.bmp, *.dib��8λ����ͨ����3ͨ����4ͨ�����롣
	// Portable image format - *.pbm, *.pgm, *.ppm��NetPBM��8λ����ͨ����PGM����3ͨ����PPM����
	// PNG - *.png��8λ��16λ����ͨ����3ͨ����4ͨ�����롣
	// Sun rasters - *.sr, *.ras
	// JPEG files - *.jpeg, *.jpg, *.jpe������JPEG��8λ����ͨ����3ͨ�����롣
	// JPEG 2000 files - *.jp2��8λ��16λ����ͨ����3ͨ�����롣
	// WebP - *.webp
	// TIFF files - *.tiff, *.tif��8λ��16λ����ͨ����3ͨ����4ͨ�����롣
	// 2��enum cv::ImreadModes ö������˵��
	// IMREAD_UNCHANGED��������ת�����籣��Ϊ16λ��ͼƬ������������16λͼƬ��
	// IMREAD_GRAYSCAL��ת��Ϊ�Ҷ�ͼ���籣��Ϊ16λͼƬ����������8λͼ��������CV_8UC1��
	// IMREAD_COLOR��ת��Ϊ��ͨ��ͼ��
	// IMREAD_ANYDEPTH����ͼ�����Ϊ16/32λ������������16/32λ��������ת��Ϊ8λ��
	// IMREAD_ANYCOLOR����ȡ��ͨ���ɾ����ļ����������3ͨ����
	// IMREAD_LOAD_GDAL��ʹ��GDAL������ȡ�ļ���GDAL��Geospatial Data Abstraction Library����X/MIT���Э���µĿ�Դդ��ռ�����ת���⡣
	//                   �����ó�������ģ���������֧�ֵĸ����ļ���ʽ��������һϵ�������й�������������ת���ʹ���
  cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
  cv::Mat img_gray;			// �Ҷ�ͼ

  if (img.channels() != 1)		// ��ͨ��������1������Ҫͨ�� cv::cvtColor() ��ת��
	  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  else                            // ��ͨ����Ϊ1�����������ͼ������ǻҶ�ͼ��ֱ�Ӹ�ֵ���� 
	  img_gray = img;

  seeta::ImageData img_data;
  img_data.data = img_gray.data;		// �Ҷ�ͼ�����ݸ�ֵ
  img_data.width = img_gray.cols;		// �Ҷ�ͼ����������Ϊͼ��Ŀ��
  img_data.height = img_gray.rows;	// �Ҷ�ͼ����������Ϊͼ��ĸ߶�
  img_data.num_channels = 1;			// �Ҷ�ͼ��ͨ�����̶�Ϊ1

  cv::imshow("�Ҷ�ͼ", img_gray);

  long t0 = cv::getTickCount();

  // �������
  // ��⵽�����������Ѿ��洢�� FaceInfo.bbox ��
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

  // �����⵽��������
  cout << "Face size:" << faces.size() << endl;

  // ����⵽����������
  cv::Rect face_rect;
  // size_t ��̬����ת��Ϊ int32_t ����
  int32_t num_face = static_cast<int32_t>(faces.size());

  for (int32_t i = 0; i < num_face; i++) {
	// ��ȡ���ƾ�������Ҫ�Ĳ�����ԭ�㣨���Ͻǣ����ꡢ�߶ȡ���ȡ�
    face_rect.x = faces[i].bbox.x;
    face_rect.y = faces[i].bbox.y;
    face_rect.width = faces[i].bbox.width;
    face_rect.height = faces[i].bbox.height;

	// ���ƾ���
    cv::rectangle(img, face_rect, CV_RGB(0, 255, 255), 1, 8, 0);
  }

  cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);

  // imshow��ʾ��ͼ������������CV_16U��ʾ16Ϊ�޷���������CV_32S��ʾ32Ϊ�з������������ɣ�
  // ͼ��ÿ�����ص�ֵ�ᱻ����256���Ա��ܹ���256���Ҷ�����ʾ������ʾ�ɸ��������ɵ�ͼ��ʱ��
  // ֵ�ķ�Χ�ᱻ����Ϊ0.0(��ɫ)��1.0����ɫ��֮�䣬����1.0��ʾ��ɫ��С��0.0��ʾ��ɫ��
  cv::imshow("Test", img);
  cv::waitKey(0);
  cv::destroyAllWindows();
}
