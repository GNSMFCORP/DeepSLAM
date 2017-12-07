#Creating lmdb for NYU-v2

Generates images and depth maps as two different LMDB files that should be loaded separately during training:

	1- create_lmdb_separated_depth: Generate LMDBs without croping
	2- create_lmdb_separated_depth_crop: Generate LMDBs with croping

Generates images and depth maps in two different files:

	- create_lmdb_separated_depth_crop_toImage
