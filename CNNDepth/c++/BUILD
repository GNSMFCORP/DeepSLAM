cc_binary(
	name = "main",
	srcs = ["main.cc"],
	linkopts = ["-lopencv_core", "-lopencv_imgproc", "-lopencv_highgui"],
	copts = ["/usr/local/include"],
	deps = [
		"//tensorflow/cc:cc_ops",
		"//tensorflow/cc/saved_model:tag_constants",
		"//tensorflow/cc/saved_model:loader_lite",
		#"//tensorflow/core:framework_internal",
		"//tensorflow/core:tensorflow",
		#"@opencv//:opencv",
	],
)