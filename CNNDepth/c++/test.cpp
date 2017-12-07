// -lopencv_core -lopencv_highgui -lopencv_imgproc
#include <utility>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

bool ReadTensorFromImageFile(const char* file_name, const int input_height,
                               const int input_width, const int input_channels,
                               cv::Mat &image) {
  std::cout << file_name << std::endl;
  cv::Mat image_in, image_resize;
  image_in = cv::imread(file_name, CV_LOAD_IMAGE_COLOR | CV_8UC3);
  cv::resize(image_in, image_resize, cv::Size(input_width, input_height));
  cv::cvtColor(image_resize, image, CV_BGR2RGB);
  
  for (int x = 0; x < input_width; x++) {
    for (int y = 0; y < input_height; y++) {
      for (int c = 0; c < input_channels; c++) {
	    image.at<cv::Vec3b>(y, x)[c] -= 1;
      }
    }
  }
  //out_image->push_back(image);  	  
}


int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cout << "Need path for an input image\n";
	}
	char* file_name = argv[1];
	int input_height = 228;
	int input_width = 304;
	int input_channels = 3;


	cv::Mat image;
	ReadTensorFromImageFile(file_name, input_height, input_width, input_channels, image);
	

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Diplay window", image);

	cv::waitKey(0);

	cv::destroyWindow("Display window");
	return 0;
}