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

	// @todo Impl �� �� FaceDetection �ĸ��࣬���䶨������ FaceDetection �ж���ģ�face_detection.cpp:44 - face_detection.cpp:74����
	class FaceDetection::Impl {
	public:
		Impl()
			: detector_(new seeta::fd::FuStDetector()),
			slide_wnd_step_x_(4), slide_wnd_step_y_(4),
			min_face_size_(20), max_face_size_(-1),
			cls_thresh_(3.85f) {}

		~Impl() {}

		// ������������
		// �ж�������Ƿ��ǺϷ���ͼƬ
		// �ж����ݣ�ͨ����Ϊ1��data��ǿգ�ͼ��ߡ����0��
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

		// unique_ptr���жԶ���Ķ���Ȩ��ͬһʱ��ֻ����һ��unique_ptrָ���������ͨ����ֹ�������塢ֻ���ƶ�������ʵ�֣���
		// unique_ptrָ�뱾����������ڣ���unique_ptrָ�봴��ʱ��ʼ��ֱ���뿪������
		// �뿪������ʱ������ָ�����������ָ��������(Ĭ��ʹ��delete���������û���ָ����������)��
		// unique_ptr<seeta::fd::Detector> detector_2 = detector_;						// err, ����ͨ������
		// std::unique_ptr<seeta::fd::Detector> detector_3 = std::move(detector_);		// ���� detector_3 ������Ψһ��unique_ptr
		std::unique_ptr<seeta::fd::Detector> detector_;			// ָ�������Ķ�ռָ��

		seeta::fd::ImagePyramid img_pyramid_;						// ͼ�������
	};

	// ���ؼ��ģ���ļ�
	FaceDetection::FaceDetection(const char* model_path)
		: impl_(new seeta::FaceDetection::Impl()) {
		impl_->detector_->LoadModel(model_path);
	}

	// ��������
	FaceDetection::~FaceDetection() {
		if (impl_ != nullptr)
			delete impl_;
	}

	// �������
	std::vector<seeta::FaceInfo> FaceDetection::Detect(
		const seeta::ImageData & img) {

		// �ж�������Ƿ��ǺϷ��ĻҶ�ͼƬ
		// ��Ա impl_ ���� FaceDetection ��������face_detection.h)
		if (!impl_->IsLegalImage(img))
			return std::vector<seeta::FaceInfo>();

		// ��СͼƬ��С
		// ���û��Զ����min_img_size��ͼ���ȡ�ͼ��߶ȣ�����ѡ��С���Ǹ���Ϊ��СͼƬ��С
		int32_t min_img_size = img.height <= img.width ? img.height : img.width;

		min_img_size = (impl_->max_face_size_ > 0 ?
			(min_img_size >= impl_->max_face_size_ ? impl_->max_face_size_ : min_img_size) :
			min_img_size);

		// ����ͼ���������ʼ��С ��
		impl_->img_pyramid_.SetImage1x(img.data, img.width, img.height);

		// ����ͼ���������С�ı�����
		// static_cast<type-id> expression ��4���÷�
		// (1) ���ڻ�����������֮���ת�������intת��Ϊchar����intת����enum��������ת���İ�ȫ����Ҫ�������Լ���֤����������Ϊ��֤���ݵľ��ȣ�������Ա�ܲ��ܱ�֤�Լ���Ҫ�ĳ���ȫ����
		//     ���ڰ�intת��Ϊcharʱ�����charû���㹻�ı���λ�����int��ֵ��int>127��int<-127ʱ������ôstatic_cast������ֻ�Ǽ򵥵Ľضϣ����򵥵ذ�int�ĵ�8λ���Ƶ�char��8λ�У���ֱ��������λ��
		// (2) �ѿ�ָ��ת����Ŀ�����͵Ŀ�ָ��
		// (3) ���κ����͵ı��ʽ����ת����void����
		// (4) �������νṹ�и��������֮��ָ������õ�ת����
		// �������ϵڣ�4���㣬����������ʽ��ת����������ת�������ൽ���ࣩ������ת�������ൽ���ࣩ������static_cast������ת��ʱ��ȫ�ģ�������ת��ʱ����ȫ�ģ�Ϊʲô�أ�
		// ��Ϊstatic_cast��ת���Ǵֱ��ģ�������������ת��������ṩ����Ϣ���������е����ͣ�������ת��������ת����ʽ��������ת���������������ǰ���������������ݳ�Ա�ͺ�����Ա��
		// ��˴�����ת���������ָ��������û���κι��ǵķ����䣨ָ���ࣩ�ĳ�Ա������������ת��Ϊʲô����ȫ������Ϊstatic_castֻ���ڱ���ʱ�������ͼ�飬û������ʱ�����ͼ�飬����ԭ����dynamic_cast��˵����
		impl_->img_pyramid_.SetMinScale(static_cast<float>(impl_->kWndSize) / min_img_size);
		
		// ���ô��ڴ�С
		impl_->detector_->SetWindowSize(impl_->kWndSize);

		// ���û������ڲ���
		impl_->detector_->SetSlideWindowStep(impl_->slide_wnd_step_x_,
			impl_->slide_wnd_step_y_);

		// ִ��ʵ���������
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
