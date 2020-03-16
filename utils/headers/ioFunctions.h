#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <ctime>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "ItkImageIO.h"

namespace fs = boost::filesystem;

bool get_data_list(std::string list_path, std::list<std::string> &filename_list) 
{
	/*
	*  @list_path: input path to file that contained file list
	*  @filename_list: data list that contained file list
	*/
	std::string filename_tmp;
	std::ifstream ifs(list_path);
	if (ifs.fail()) {
		std::cerr << "Cannot open file: " << list_path << std::endl;
		return false;
	}
	while (getline(ifs, filename_tmp)) {
		filename_list.push_back(filename_tmp);
	}
	ifs.close();

	return true;
}

template<class T>
void save_vector(const std::string path, std::vector<T> img,
	const int xsize, const int ysize, const int zsize,
	const double xspacing, const double yspacing, const double zspacing)
{
	ImageIO<3> save_mhd;
	save_mhd.SetIndex(0, 0);
	save_mhd.SetIndex(1, 0);
	save_mhd.SetIndex(2, 0);
	save_mhd.SetSize(0, xsize);
	save_mhd.SetSize(1, ysize);
	save_mhd.SetSize(2, zsize);
	save_mhd.SetOrigin(0, 0.);
	save_mhd.SetOrigin(1, 0.);
	save_mhd.SetOrigin(2, 0.);
	save_mhd.SetSpacing(0, xspacing);
	save_mhd.SetSpacing(1, yspacing);
	save_mhd.SetSpacing(2, zspacing);
	save_mhd.Write(img, path);
	return;
}

bool make_dir(std::string dir_path) 
{
	/*
	* https://boostjp.github.io/tips/filesystem.html
	*/
	const fs::path path(dir_path);
	boost::system::error_code error;
	const bool result = fs::create_directories(path, error);
	if (!result || error) {
		return false;
	}
	return true;
}

std::string get_parent_path(const std::string file_path)
{
	/*
	*  https://hwada.hatenablog.com/entry/20110611/1307781684
	*/
	fs::path path(file_path);
	return path.parent_path().string();
}

std::string get_dirname(const std::string dirpath)
{
	/*
	*  https://hwada.hatenablog.com/entry/20110611/1307781684
	*/
	std::string delim("\\/");
	std::list<std::string> results;
	boost::split(results, dirpath, boost::is_any_of(delim));
	return *std::next(results.begin(), results.size()-1);
}

std::string get_filename(const std::string file_path)
{
	/*
	*  https://hwada.hatenablog.com/entry/20110611/1307781684
	*/
	fs::path path(file_path);
	return path.filename().string();
}

std::string get_stem(const std::string file_path)
{
	/*
	*  https://hwada.hatenablog.com/entry/20110611/1307781684
	*/
	fs::path path(file_path);
	return path.stem().string();
}

std::string get_local_now()
{
	// https://dianxnao.com/c%E5%88%9D%E7%B4%9A%EF%BC%9A%E6%97%A5%E4%BB%98%E3%81%A8%E6%99%82%E5%88%BB%E3%82%92%E5%8F%96%E5%BE%97%E3%81%99%E3%82%8B/

	time_t now = time(nullptr);
	struct tm local_now;
	localtime_s(&local_now, &now);

	std::string year = std::to_string(local_now.tm_year + 1900);
	std::string month = std::to_string(local_now.tm_mon + 1);
	std::string day = std::to_string(local_now.tm_mday);
	std::string hour = std::to_string(local_now.tm_hour);
	std::string minute = std::to_string(local_now.tm_min);
	std::string second = std::to_string(local_now.tm_sec);

	return (year + "-" + month + "-" + day + "_" + hour + "-" + minute + "-" + second);
}

void save_args(int argc, char **argv, std::string result_dir)
{
	
	// const int argc = sizeof(argv[0]) / sizeof(argv[0][0]);
	std::string now = get_local_now();
	const std::string filename = "/configs_" + now + ".txt";
	make_dir(result_dir);
	const std::string output_filename = result_dir + filename;
	
	std::ofstream output(output_filename);
	for (int i = 0; i < argc; i++) {
		output << argv[i] << std::endl;
	}
	output.close();

}

template<class InputPixelType>
int save_img(std::string path, typename itk::Image<InputPixelType, 3>::Pointer ptr)
{
	//using ImageType = itk::Image<double, 3>;
	using ImageType = itk::Image<InputPixelType, 3>;
	using WriterType = itk::ImageFileWriter< ImageType >;
	WriterType::Pointer writer = WriterType::New();
	writer->SetInput(ptr);
	writer->SetFileName(path);
	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject & error)
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

template<class T>
void save_vector_to_csv(std::string file_path, std::vector<T> vect)
{

	std::ofstream ofs(file_path);
	for (auto itr = vect.begin(); itr != vect.end(); itr++) {
		ofs << *itr << std::endl;
	}
	ofs.close();

	return;
}