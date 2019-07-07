/*
* Make train hr patch
* @auther tozawa
* @date 20181116
*/

#include<iostream>
#include<filesystem>
#include <fstream>
#include"ItkImageIO.h"

typedef long long llong;
typedef unsigned char in_type;

// Patch side and patch size 
int patch_side;
int patch_size; // = patch_side * patch_side * patch_side

// Sampling interval in input image
int interval;

std::string make_filename(const char * inputFileName, const char * outputFileDir, const int total) {

	std::tr2::sys::path path(inputFileName);
	std::string save_path = std::string(outputFileDir) + "/" + path.stem().string()+ "_total" + std::to_string(total) 
		 + "_interval" + std::to_string(interval) + ".csv";
	std::cout << "outputFile: " << save_path << std::endl;

	return save_path;
}

int main(int argc, char **argv)
{
	if (argc != 6) {
		std::cerr << "Usage:" << std::endl;
		std::cerr << argv[0] << " inputFile(.mhd) maskFile(.mhd) outputDir patchSide samplingInterval" << std::endl;
		exit(1);
	}

	const char *inputFileName = argv[1];
	const char *maskFileName = argv[2];
	const char *outputDir = argv[3];
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
		for (llong z = 0; z < ze-patch_side; z += interval) {
			for (llong y = 0; y < ye - patch_side; y += interval) {
				for (llong x = 0; x < xe - patch_side; x += interval) {
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
		std::ofstream log;
		std::string save_path = make_filename(inputFileName, outputDir, patchCount);
		log.open(save_path);

		/// Raster Scan
		for (llong z = 0; z < ze - patch_side; z += interval) {
			for (llong y = 0; y < ye - patch_side; y += interval) {
				for (llong x = 0; x < xe - patch_side; x += interval) {
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
						count++;
						log << x << "," << y << "," << z << std::endl;
					}

				}
			}
		}/*Raster Scan*/

		log.close();
		assert(count == patchCount);

	}

	return 0;
}