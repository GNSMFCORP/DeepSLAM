#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const int input_channels,
                               std::vector<Tensor>* out_tensors) {

  cv::Mat image_in, image_resize, image;
  image_in = cv::imread(file_name);
  cv::resize(image_in, image_resize, cv::Size(input_width, input_height));
  cv::cvtColor(image_resize, image, CV_BGR2RGB);

  Tensor output_tensor(tensorflow::DT_FLOAT,
  		tensorflow::TensorShape({1, input_width, input_height, input_channels}));

  auto output_tensor_mapped = output_tensor.tensor<float, 4>();
  for (int x = 0; x < input_width; x++) {
  	for (int y = 0; y < input_height; y++) {
  	  for (int c = 0; c < input_channels; c++) {
  	  	output_tensor_mapped(0, x, y, c) = (float)image.at<cv::Vec3b>(y, x)[c];
  	  }  	  
  	}
  }

  out_tensors->push_back(output_tensor);

  return Status::OK();
}

Status TensorToImage(const Tensor input_tensor, const int input_height,
                     const int input_width, cv::Mat &image) {
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();
  for (int x = 0; x < input_width; x++) {
    for (int y = 0; y < input_height; y++) {
  	  image.at<float>(y, x) = input_tensor_mapped(0, y, x, 0); 	  
  	}
  }
  return Status::OK();
}


int main(int argc, char* argv[]) {
	int32 input_height = 228;
	int32 input_width = 304;
	int32 input_channels = 3;
	if (argc < 3) {
		LOG(ERROR) << "input format: image_path graph_path";
	}
	string image_path = argv[1];
  string graph_path = argv[2];
	string input_layer = "Placeholder:0";
	string output_layer = "ConvPred/ConvPred:0";

	tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::SavedModelBundle modelBundle;
  Status load_status = tensorflow::LoadSavedModel(tensorflow::SessionOptions(),
                        tensorflow::RunOptions(), graph_path, 
                        {tensorflow::kSavedModelTagServe}, &modelBundle);
  if (!load_status.ok()) {
    LOG(ERROR) << load_status;
    return -1;
  }

	std::vector<Tensor> resized_tensors;
	Status read_tensor_status = 
		ReadTensorFromImageFile(image_path, input_height, input_width, 
								input_channels, &resized_tensors);
	if (!read_tensor_status.ok()) {
		LOG(ERROR) << read_tensor_status;
		return -1;
	}
	const Tensor& resized_tensor = resized_tensors[0];

  
	std::vector<Tensor> outputs;
	Status run_status = modelBundle.session->Run({{input_layer, resized_tensor}},
									 {output_layer}, {}, &outputs);
	if (!run_status.ok()) {
		LOG(ERROR) << "Running model failed: " << run_status;
    	return -1;
	}  

	cv::Mat image(128, 160, CV_32FC1);
	TensorToImage(outputs[0], 128, 160, image);

	return 0;
}