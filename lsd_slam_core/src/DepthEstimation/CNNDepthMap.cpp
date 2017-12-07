#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include "DepthEstimation/CNNDepthMap.h"

#include "DepthEstimation/DepthMapPixelHypothesis.h"
#include "DataStructures/Frame.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "util/EigenCoreInclude.h"
#include "IOWrapper/ImageDisplay.h"

#include "DepthEstimation/Classifier.h"
#include "util/settings.h"

namespace lsd_slam
{

string model_file= "/home/hamid/Depth-Estimation/net_deploy.prototxt";
string trained_file=  "/home/hamid/Depth-Estimation/model_norm_abs_100k.caffemodel";
Classifier classifier(model_file,trained_file);


void CNNDepthMap::updateKeyFrame(std::deque< std::shared_ptr<Frame> > referenceFrames)
{
	assert(isValid());

	struct timeval tv_start_all, tv_end_all;
	gettimeofday(&tv_start_all, NULL);

	oldest_referenceFrame = referenceFrames.front().get();
	newest_referenceFrame = referenceFrames.back().get();
	referenceFrameByID.clear();
	referenceFrameByID_offset = oldest_referenceFrame->id();

	for(std::shared_ptr<Frame> frame : referenceFrames)
	{
		assert(frame->hasTrackingParent());

		if(frame->getTrackingParent() != activeKeyFrame)
		{
			printf("WARNING: updating frame %d with %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
					activeKeyFrame->id(), frame->id(),
					frame->getTrackingParent()->id());
		}

		Sim3 refToKf;
		if(frame->pose->trackingParent->frameID == activeKeyFrame->id())
			refToKf = frame->pose->thisToParent_raw;
		else
			refToKf = activeKeyFrame->getScaledCamToWorld().inverse() *  frame->getScaledCamToWorld();

		frame->prepareForStereoWith(activeKeyFrame, refToKf, K, 0);

		while((int)referenceFrameByID.size() + referenceFrameByID_offset <= frame->id())
			referenceFrameByID.push_back(frame.get());
	}

	resetCounters();

	
	if(plotStereoImages)
	{
		cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(), CV_32F, const_cast<float*>(activeKeyFrameImageData));
		keyFrameImage.convertTo(debugImageHypothesisHandling, CV_8UC1);
		cv::cvtColor(debugImageHypothesisHandling, debugImageHypothesisHandling, CV_GRAY2RGB);

		cv::Mat oldest_refImage(oldest_referenceFrame->height(), oldest_referenceFrame->width(), CV_32F, const_cast<float*>(oldest_referenceFrame->image(0)));
		cv::Mat newest_refImage(newest_referenceFrame->height(), newest_referenceFrame->width(), CV_32F, const_cast<float*>(newest_referenceFrame->image(0)));
		cv::Mat rfimg = 0.5f*oldest_refImage + 0.5f*newest_refImage;
		rfimg.convertTo(debugImageStereoLines, CV_8UC1);
		cv::cvtColor(debugImageStereoLines, debugImageStereoLines, CV_GRAY2RGB);
	}

	struct timeval tv_start, tv_end;


	gettimeofday(&tv_start, NULL);
	observeDepth();
	gettimeofday(&tv_end, NULL);
	msObserve = 0.9*msObserve + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nObserve++;

	//if(rand()%10==0)
	{
		gettimeofday(&tv_start, NULL);
		regularizeDepthMapFillHoles();
		gettimeofday(&tv_end, NULL);
		msFillHoles = 0.9*msFillHoles + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nFillHoles++;
	}


	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9*msRegularize + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;

	
	// Update depth in keyframe
	if(!activeKeyFrame->depthHasBeenUpdatedFlag)
	{
		gettimeofday(&tv_start, NULL);
		activeKeyFrame->setDepth(currentDepthMap);
		gettimeofday(&tv_end, NULL);
		msSetDepth = 0.9*msSetDepth + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nSetDepth++;
	}


	gettimeofday(&tv_end_all, NULL);
	msUpdate = 0.9*msUpdate + 0.1*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nUpdate++;


	activeKeyFrame->numMappedOnThis++;
	activeKeyFrame->numMappedOnThisTotal++;


	if(plotStereoImages)
	{
		Util::displayImage( "Stereo Key Frame", debugImageHypothesisHandling, false );
		Util::displayImage( "Stereo Reference Frame", debugImageStereoLines, false );
	}



	if(enablePrintDebugInfo && printLineStereoStatistics)
	{
		printf("ST: calls %6d, comp %6d, int %7d; good %6d (%.0f%%), neg %6d (%.0f%%); interp %6d / %6d / %6d\n",
				runningStats.num_stereo_calls,
				runningStats.num_stereo_comparisons,
				runningStats.num_pixelInterpolations,
				runningStats.num_stereo_successfull,
				100*runningStats.num_stereo_successfull / (float) runningStats.num_stereo_calls,
				runningStats.num_stereo_negative,
				100*runningStats.num_stereo_negative / (float) runningStats.num_stereo_successfull,
				runningStats.num_stereo_interpPre,
				runningStats.num_stereo_interpNone,
				runningStats.num_stereo_interpPost);
	}
	if(enablePrintDebugInfo && printLineStereoFails)
	{
		printf("ST-ERR: oob %d (scale %d, inf %d, near %d); err %d (%d uncl; %d end; zro: %d btw, %d no, %d two; %d big)\n",
				runningStats.num_stereo_rescale_oob+
					runningStats.num_stereo_inf_oob+
					runningStats.num_stereo_near_oob,
				runningStats.num_stereo_rescale_oob,
				runningStats.num_stereo_inf_oob,
				runningStats.num_stereo_near_oob,
				runningStats.num_stereo_invalid_unclear_winner+
					runningStats.num_stereo_invalid_atEnd+
					runningStats.num_stereo_invalid_inexistantCrossing+
					runningStats.num_stereo_invalid_noCrossing+
					runningStats.num_stereo_invalid_twoCrossing+
					runningStats.num_stereo_invalid_bigErr,
				runningStats.num_stereo_invalid_unclear_winner,
				runningStats.num_stereo_invalid_atEnd,
				runningStats.num_stereo_invalid_inexistantCrossing,
				runningStats.num_stereo_invalid_noCrossing,
				runningStats.num_stereo_invalid_twoCrossing,
				runningStats.num_stereo_invalid_bigErr);
	}

}


void CNNDepthMap::createKeyFrame(Frame* new_keyframe)
{
	assert(isValid());
	assert(new_keyframe != nullptr);
	assert(new_keyframe->hasTrackingParent());

	//boost::shared_lock<boost::shared_mutex> lock = activeKeyFrame->getActiveLock();
	boost::shared_lock<boost::shared_mutex> lock2 = new_keyframe->getActiveLock();

	struct timeval tv_start_all, tv_end_all;
	gettimeofday(&tv_start_all, NULL);


	resetCounters();

	if(plotStereoImages)
	{
		cv::Mat keyFrameImage(new_keyframe->height(), new_keyframe->width(), CV_32F, const_cast<float*>(new_keyframe->image(0)));
		keyFrameImage.convertTo(debugImageHypothesisPropagation, CV_8UC1);
		cv::cvtColor(debugImageHypothesisPropagation, debugImageHypothesisPropagation, CV_GRAY2RGB);
	}



	SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();

	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);
	getCNNDepth(new_keyframe);
	gettimeofday(&tv_end, NULL);
	msPropagate = 0.9*msPropagate + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nPropagate++;

	activeKeyFrame = new_keyframe;
	activeKeyFramelock = activeKeyFrame->getActiveLock();
	activeKeyFrameImageData = new_keyframe->image(0);
	activeKeyFrameIsReactivated = false;



	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(true, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9*msRegularize + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;


	gettimeofday(&tv_start, NULL);
	regularizeDepthMapFillHoles();
	gettimeofday(&tv_end, NULL);
	msFillHoles = 0.9*msFillHoles + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nFillHoles++;


	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9*msRegularize + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;




	// make mean inverse depth be one.
	float sumIdepth=0, numIdepth=0;
	for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+width*height; source++)
	{
		if(!source->isValid)
			continue;
		sumIdepth += source->idepth_smoothed;
		numIdepth++;
	}
	float rescaleFactor = numIdepth / sumIdepth;
	float rescaleFactor2 = rescaleFactor*rescaleFactor;
	for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+width*height; source++)
	{
		if(!source->isValid)
			continue;
		source->idepth *= rescaleFactor;
		source->idepth_smoothed *= rescaleFactor;
		source->idepth_var *= rescaleFactor2;
		source->idepth_var_smoothed *= rescaleFactor2;
	}
	activeKeyFrame->pose->thisToParent_raw = sim3FromSE3(oldToNew_SE3.inverse(), rescaleFactor);
	activeKeyFrame->pose->invalidateCache();

	// Update depth in keyframe

	gettimeofday(&tv_start, NULL);
	activeKeyFrame->setDepth(currentDepthMap);
	gettimeofday(&tv_end, NULL);
	msSetDepth = 0.9*msSetDepth + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nSetDepth++;

	gettimeofday(&tv_end_all, NULL);
	msCreate = 0.9*msCreate + 0.1*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nCreate++;



	if(plotStereoImages)
	{
		//Util::displayImage( "KeyFramePropagation", debugImageHypothesisPropagation );
	}

}

void CNNDepthMap::finalizeKeyframe()
{
	assert(isValid());


	struct timeval tv_start_all, tv_end_all;
	gettimeofday(&tv_start_all, NULL);
	struct timeval tv_start, tv_end;

	gettimeofday(&tv_start, NULL);
	regularizeDepthMapFillHoles();
	gettimeofday(&tv_end, NULL);
	msFillHoles = 0.9*msFillHoles + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nFillHoles++;

	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9*msRegularize + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;

	gettimeofday(&tv_start, NULL);
	activeKeyFrame->setDepth(currentDepthMap);
	activeKeyFrame->calculateMeanInformation();
	activeKeyFrame->takeReActivationData(currentDepthMap);
	gettimeofday(&tv_end, NULL);
	msSetDepth = 0.9*msSetDepth + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nSetDepth++;

	gettimeofday(&tv_end_all, NULL);
	msFinalize = 0.9*msFinalize + 0.1*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nFinalize++;
}


void CNNDepthMap::setFromExistingKF(Frame* kf)
{
	assert(kf->hasIDepthBeenSet());

	activeKeyFramelock = kf->getActiveLock();
	activeKeyFrame = kf;

	const float* idepth = activeKeyFrame->idepth_reAct();
	const float* idepthVar = activeKeyFrame->idepthVar_reAct();
	const unsigned char* validity = activeKeyFrame->validity_reAct();

	DepthMapPixelHypothesis* pt = currentDepthMap;
	activeKeyFrame->numMappedOnThis = 0;
	activeKeyFrame->numFramesTrackedOnThis = 0;
	activeKeyFrameImageData = activeKeyFrame->image(0);
	activeKeyFrameIsReactivated = true;

	for(int y=0;y<height;y++)
	{
		for(int x=0;x<width;x++)
		{
			if(*idepthVar > 0)
			{
				*pt = DepthMapPixelHypothesis(
						*idepth,
						*idepthVar,
						*validity);
			}
			else
			{
				currentDepthMap[x+y*width].isValid = false;
				currentDepthMap[x+y*width].blacklisted = (*idepthVar == -2) ? MIN_BLACKLIST-1 : 0;
			}

			idepth++;
			idepthVar++;
			validity++;
			pt++;
		}
	}

	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);

}

void CNNDepthMap::getCNNDepth(Frame* new_keyframe)
{
	if(new_keyframe->getTrackingParent() != activeKeyFrame)
	{
		printf("WARNING: propagating depth from frame %d to %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
				activeKeyFrame->id(), new_keyframe->id(),
				new_keyframe->getTrackingParent()->id());
	}

	// wipe depthmap
	for(DepthMapPixelHypothesis* pt = otherDepthMap+width*height-1; pt >= otherDepthMap; pt--)
	{
		pt->isValid = false;
		pt->blacklisted = 0;
	}

	cv::Mat CNN_input(new_keyframe->height(), activeKeyFrame->width(), CV_32F, const_cast<float*>(new_keyframe->image(0)));
  	//cv::imwrite("/home/hamid/slam/image_CNN.jpg",CNN_input);
	//keyFrameImage.convertTo(debugImageDepth, CV_8UC1);
	cv::cvtColor(CNN_input, CNN_input, CV_GRAY2RGB);
	// Passing through CNN
	cv::Mat img_resized;
	cv::resize(CNN_input,img_resized,cv::Size(298,218));
	Blob<float>* prediction = classifier.Predict(img_resized);
	float* data_CNN = prediction->mutable_cpu_data();
    //resizing the output
	cv::Size ss(74,54);
	cv::Mat depth_CNN(ss, CV_32FC1, data_CNN);
	depth_CNN.setTo(1.0, depth_CNN>1);
	depth_CNN.setTo(0.0, depth_CNN<0); 
	std::cout << "INJIII" << std::endl;
	cv::Mat depth_resized;
	cv::resize(depth_CNN,depth_resized,cv::Size(640,480));
//	cv::imwrite("/home/hamid/slam/depth_CNN.jpg",depth_resized*255);

	
	for (int x = 0; x < width; x ++)
		for (int y = 0; y < height; y ++) {
			int idx=x+y*width;
			DepthMapPixelHypothesis* ptr_dst = otherDepthMap+idx;
			otherDepthMap[idx] = DepthMapPixelHypothesis(
					depth_resized.at<float>(y, x)*8,
					1.0f,
					VALIDITY_COUNTER_INITIAL_OBSERVE);
			otherDepthMap[x+y*width].isValid=true;
		}
	std::swap(currentDepthMap, otherDepthMap);
}



}
