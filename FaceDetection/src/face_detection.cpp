/*
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is part of the SeetaFace Detection module, containing codes implementing the
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

#include "face_detection.h"

#include <memory>
#include <vector>

#include "detector.h"
#include "fust.h"
#include "util/image_pyramid.h"

namespace seeta {

	// @todo Impl 是 类 FaceDetection 的父类，但其定义是在 FaceDetection 中定义的（face_detection.cpp:44 - face_detection.cpp:74）？
	class FaceDetection::Impl {
	public:
		Impl()
			: detector_(new seeta::fd::FuStDetector()),
			slide_wnd_step_x_(4), slide_wnd_step_y_(4),
			min_face_size_(20), max_face_size_(-1),
			cls_thresh_(3.85f) {}

		~Impl() {}

		// 类中内联函数
		// 判断输入的是否是合法的图片
		// 判断依据：通道数为1，data域非空，图像高、宽非0。
		inline bool IsLegalImage(const seeta::ImageData & image) {
			return (image.num_channels == 1 && image.width > 0 && image.height > 0 &&
				image.data != nullptr);
		}

	public:
		static const int32_t kWndSize = 40;

		int32_t min_face_size_;
		int32_t max_face_size_;
		int32_t slide_wnd_step_x_;
		int32_t slide_wnd_step_y_;
		float cls_thresh_;

		std::vector<seeta::FaceInfo> pos_wnds_;

		// unique_ptr持有对对象的独有权，同一时刻只能有一个unique_ptr指向给定对象（通过禁止拷贝语义、只有移动语义来实现）。
		// unique_ptr指针本身的生命周期：从unique_ptr指针创建时开始，直到离开作用域。
		// 离开作用域时，若其指向对象，则将其所指对象销毁(默认使用delete操作符，用户可指定其他操作)。
		// unique_ptr<seeta::fd::Detector> detector_2 = detector_;						// err, 不能通过编译
		// std::unique_ptr<seeta::fd::Detector> detector_3 = std::move(detector_);		// 现在 detector_3 是数据唯一的unique_ptr
		std::unique_ptr<seeta::fd::Detector> detector_;			// 指向检测器的独占指针

		seeta::fd::ImagePyramid img_pyramid_;						// 图像金字塔
	};

	// 加载检测模型文件
	FaceDetection::FaceDetection(const char* model_path)
		: impl_(new seeta::FaceDetection::Impl()) {
		impl_->detector_->LoadModel(model_path);
	}

	// 析构函数
	FaceDetection::~FaceDetection() {
		if (impl_ != nullptr)
			delete impl_;
	}

	// 人脸检测
	std::vector<seeta::FaceInfo> FaceDetection::Detect(
		const seeta::ImageData & img) {

		// 判断输入的是否是合法的灰度图片
		// 成员 impl_ 在类 FaceDetection 中声明（face_detection.h)
		if (!impl_->IsLegalImage(img))
			return std::vector<seeta::FaceInfo>();

		// 最小图片大小
		// 从用户自定义的min_img_size、图像宽度、图像高度，三中选最小的那个作为最小图片大小
		int32_t min_img_size = img.height <= img.width ? img.height : img.width;

		min_img_size = (impl_->max_face_size_ > 0 ?
			(min_img_size >= impl_->max_face_size_ ? impl_->max_face_size_ : min_img_size) :
			min_img_size);

		// 设置图像金字塔初始大小 ？
		impl_->img_pyramid_.SetImage1x(img.data, img.width, img.height);

		// 设置图像金字塔最小的比例尺
		// static_cast<type-id> expression 的4种用法
		// (1) 用于基本数据类型之间的转换，如把int转换为char，把int转换成enum，但这种转换的安全性需要开发者自己保证（这可以理解为保证数据的精度，即程序员能不能保证自己想要的程序安全），
		//     如在把int转换为char时，如果char没有足够的比特位来存放int的值（int>127或int<-127时），那么static_cast所做的只是简单的截断，及简单地把int的低8位复制到char的8位中，并直接抛弃高位。
		// (2) 把空指针转换成目标类型的空指针
		// (3) 把任何类型的表达式类型转换成void类型
		// (4) 用于类层次结构中父类和子类之间指针和引用的转换。
		// 对于以上第（4）点，存在两种形式的转换，即上行转换（子类到父类）和下行转换（父类到子类）。对于static_cast，上行转换时安全的，而下行转换时不安全的，为什么呢？
		// 因为static_cast的转换是粗暴的，它仅根据类型转换语句中提供的信息（尖括号中的类型）来进行转换，这种转换方式对于上行转换，由于子类总是包含父类的所有数据成员和函数成员，
		// 因此从子类转换到父类的指针对象可以没有任何顾虑的访问其（指父类）的成员。而对于下行转换为什么不安全，是因为static_cast只是在编译时进行类型检查，没有运行时的类型检查，具体原理在dynamic_cast中说明。
		impl_->img_pyramid_.SetMinScale(static_cast<float>(impl_->kWndSize) / min_img_size);
		
		// 设置窗口大小
		impl_->detector_->SetWindowSize(impl_->kWndSize);

		// 设置滑动窗口步距
		impl_->detector_->SetSlideWindowStep(impl_->slide_wnd_step_x_,
			impl_->slide_wnd_step_y_);

		// 执行实际人脸检测
		impl_->pos_wnds_ = impl_->detector_->Detect(&(impl_->img_pyramid_));

		for (int32_t i = 0; i < impl_->pos_wnds_.size(); i++) {
			if (impl_->pos_wnds_[i].score < impl_->cls_thresh_) {
				impl_->pos_wnds_.resize(i);
				break;
			}
		}

		return impl_->pos_wnds_;
	}

	void FaceDetection::SetMinFaceSize(int32_t size) {
		if (size >= 20) {
			impl_->min_face_size_ = size;
			impl_->img_pyramid_.SetMaxScale(impl_->kWndSize / static_cast<float>(size));
		}
	}

	void FaceDetection::SetMaxFaceSize(int32_t size) {
		if (size >= 0)
			impl_->max_face_size_ = size;
	}

	void FaceDetection::SetImagePyramidScaleFactor(float factor) {
		if (factor >= 0.01f && factor <= 0.99f)
			impl_->img_pyramid_.SetScaleStep(static_cast<float>(factor));
	}

	void FaceDetection::SetWindowStep(int32_t step_x, int32_t step_y) {
		if (step_x > 0)
			impl_->slide_wnd_step_x_ = step_x;
		if (step_y > 0)
			impl_->slide_wnd_step_y_ = step_y;
	}

	void FaceDetection::SetScoreThresh(float thresh) {
		if (thresh >= 0)
			impl_->cls_thresh_ = thresh;
	}

}  // namespace seeta
