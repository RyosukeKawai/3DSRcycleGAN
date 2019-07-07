/*
* Make train hr patch 
* @auther tozawa
* @date 20181011
*/

#include<iostream>
#include<math.h>
#include<filesystem>
#include"Eigen/Core"
#include"ItkImageIO.h"
#include"function.h"

typedef long long llong;
typedef unsigned char in_type;

// number of patch in each chunk 
#define MAX 3000

// Patch side and patch size 
int patch_side;
int patch_size; // = patch_side * patch_side * patch_side

// Sampling interval in input image
int interval;

std::string make_filename(const char * outputFileName, const int num_col_count, int &num_raw_count) {

	std::tr2::sys::path path(outputFileName);
	std::string save_path = path.parent_path().string() + "/" + path.stem().string()
		+ "_col" + std::to_string(num_col_count) + "_interval" + std::to_string(interval) + "_" + std::to_string(num_raw_count++);
	std::cout << "outputFile: " << save_path << std::endl;

	return save_path;
}

int main(int argc, char **argv)
{
	if (argc != 6) {
		std::cerr << "Usage:" << std::endl;
		std::cerr << argv[0] << " inputFile(.mhd) maskFile(.mhd) outputFileName(.raw) patchSide samplingInterval" << std::endl;
		exit(1);
	}

	const char *inputFileName = argv[1];
	const char *maskFileName = argv[2];
	const char *outputFileName = argv[3];
	patch_side = std::stoi(argv[4]);
	patch_size = patch_side * patch_side * patch_side;
	interval = std::stoi(argv[5]);

	if (interval <= 0 || patch_side % 2 != 0) {
		std::cerr << "!! Warning !! " << std::endl;
		std::cerr << "PATCH_SIDE must be only even" << std::endl;
		std::cerr << "INTERVAL must be positive." << std::endl;
		exit(1);
	}
	std::cout << "Patch Side: " << patch_side << std::endl;
	std::cout << "Sampling Interval: " << interval << std::endl;

	// Read input images
	std::vector<in_type> img;
	std::vector<unsigned char> mask;
	ImageIO<3> mhdi, mhdo;
	mhdi.Read(img, inputFileName);
	mhdo.Read(mask, maskFileName);
	llong xe = mhdi.Size(0);
	llong ye = mhdi.Size(1);
	llong ze = mhdi.Size(2);
	llong se = xe*ze*ye;
	std::cout << xe << " " << ye << " " << ze << " " << se << std::endl;

	long patchCount = 0;
	{
		/// Raster Scan
		for (llong z = 0; z < ze; z += interval) {
			for (llong y = 0; y < ye; y += interval) {
				for (llong x = 0; x < xe; x += interval) {
					int i = 0;

					/// Confirm this patch whether to count it 
					for (llong zz = 0; zz < patch_side; ++zz) {
						for (llong yy = 0; yy < patch_side; ++yy) {
							for (llong xx = 0; xx < patch_side; ++xx) {
								llong ss = xe*(std::max((llong)0, std::min(ze - 1, z + zz))*ye
									+ std::max((llong)0, std::min(ye - 1, y + yy)))
									+ std::max((llong)0, std::min(xe - 1, x + xx));
								if (mask.at(ss) != 0) ++i;
								else zz = yy = xx = patch_side - 1;
							}
						}
					}/*Confirm*/

					 /// Count num of patches
					if (i == patch_size) ++patchCount;

				}
			}
		}/*Raster Scan*/
		std::cout << "# of Patches : " << patchCount << std::endl;
	}



	{
		long count = 0; /// Instead of 'patchCount'
		int num_col_count = 0;
		int num_raw_count = 0;
		int num_raw = (int)std::floor((double)patchCount / (double)MAX);
		int fraction = patchCount%MAX;
		std::cout << "# of raw: " << num_raw + 1 << std::endl;

		Eigen::MatrixXf	patchMatrix = Eigen::MatrixXf::Zero(patch_size, MAX);

		/// Raster Scan
		for (llong z = 0; z < ze; z += interval) {
			for (llong y = 0; y < ye; y += interval) {
				for (llong x = 0; x < xe; x += interval) {
					int i = 0;
					std::vector<float> patch(patch_size);

					/// Confirm this patch whether to count it 
					for (llong zz = 0; zz < patch_side; ++zz) {
						for (llong yy = 0; yy < patch_side; ++yy) {
							for (llong xx = 0; xx < patch_side; ++xx) {
								llong ss = xe*(std::max((llong)0, std::min(ze - 1, z + zz))*ye
									+ std::max((llong)0, std::min(ye - 1, y + yy)))
									+ std::max((llong)0, std::min(xe - 1, x + xx));
								if (mask.at(ss) != 0) patch.at(i++) = (float)img.at(ss);
								else zz = yy = xx = patch_side - 1;
							}
						}
					}/*Confirm*/

					 /// Save patch to Eigen matrix
					if (i == patch_size) {
						Eigen::Map<Eigen::VectorXf> patchVector(&patch[0], patch.size());
						patchMatrix.col(num_col_count++) = patchVector;
						count++;

						if (num_raw_count != num_raw && num_col_count == MAX) {
							std::cout << "Start save: " << num_raw_count << std::endl;
							std::string save_path = make_filename(outputFileName, num_col_count, num_raw_count);
							write_raw_and_txt_ColMajor(patchMatrix, save_path);
							std::cout << "Save done " << std::endl;
							num_col_count = 0;
						}
						else if (num_raw_count == num_raw && num_col_count == fraction) {
							std::cout << "Start save: " << num_raw_count << std::endl;
							std::string save_path = make_filename(outputFileName, num_col_count, num_raw_count);
							Eigen::MatrixXf	_patchMatrix = Eigen::MatrixXf::Zero(patch_size, fraction);
							_patchMatrix = patchMatrix.block(0, 0, patch_size, fraction);
							write_raw_and_txt_ColMajor(_patchMatrix, save_path);
							std::cout << "Save done " << std::endl;
							num_col_count = 0;
						}
					}

				}
			}
		}/*Raster Scan*/

		assert(count == patchCount);

	}

	return 0;
}