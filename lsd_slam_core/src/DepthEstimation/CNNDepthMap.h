#pragma once
#include "DepthEstimation/DepthMap.h"
//#include <string>

using std::string;
namespace lsd_slam
{

class Frame;
//class Classifier;

class CNNDepthMap: public DepthMap
{
public:
	CNNDepthMap(int w, int h, const Eigen::Matrix3f& K): DepthMap(w, h, K){int a=0;};//{ cnnGraph = new CNNGraph();};
	void updateKeyFrame(std::deque< std::shared_ptr<Frame> > referenceFrames);
	void createKeyFrame(Frame* new_keyframe);
	void finalizeKeyframe();
	void setFromExistingKF(Frame* kf);
private:

	//Classifier classifier;
	void getCNNDepth(Frame* keyframe);
	void resize(float* image_in, int width, int height, float* image_out);
};

}
