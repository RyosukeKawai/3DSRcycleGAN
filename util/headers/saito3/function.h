#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "Eigen/Core"




long long get_file_size(const std::string filename)
{
	/*
	ファイルのサイズを取得するプログラム
	filename : ファイル名
	*/

	FILE *fp;
	struct _stat64 st;
	if (fopen_s(&fp, filename.c_str(), "rb") != 0){
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	_fstat64(_fileno(fp), &st);
	fclose(fp);
	return st.st_size;
}

template< class T >
void read_vector(std::vector<T> &v, const std::string filename)
{
	/*
	raw画像を読み込んでvectorに格納
	v : 格納するベクター
	filename : ファイル名
	*/
	auto num = get_file_size(filename) / sizeof(T);
	FILE *fp;
	if (fopen_s(&fp, filename.c_str(), "rb") != 0){
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	v.resize(num);
	fread(v.data(), sizeof(T), num, fp);

	fclose(fp);
}

template< class T >
void write_vector(std::vector<T> &v, const std::string filename)
{
	/*
	vectorをraw画像に書き込み
	v : 格納するベクター
	filename : 保存場所絶対パス
	*/
	FILE *fp;
	if (fopen_s(&fp, filename.c_str(), "wb") != 0){
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	fwrite(v.data(), sizeof(T), v.size(), fp);
	fclose(fp);
}

template<typename T>
void load_raw_to_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& eigen, std::string filename, int row_size)
{																										
	std::vector<T> v;
	auto num = get_file_size(filename) / sizeof(T);
	FILE *fp;
	if (fopen_s(&fp, filename.c_str(), "rb") != 0){
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	v.resize(num);
	fread(v.data(), sizeof(T), num, fp);

	fclose(fp);

	size_t col_size = v.size() / row_size;

	// 早いかな～と思って変更(20171121 tozawa)
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(&v[0], col_size, row_size);
	eigen = A.transpose();
	
	/* //SaitoSan Origin code
	eigen = eigen.setZero(row_size, col_size);
	
	for (long int y = 0; y < row_size; y++)
	{
		for (long int x = 0; x < col_size; x++)
		{
			eigen(y, x) = v[y * col_size + x];
		}
	}*/
}

template<typename T>
void write_raw_and_txt(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data, std::string filename)
{
	/*
	eigenのmatrixをraw画像として保存
	出力はraw画像とサイズと型が記されたtxtファイル
	data : 保存するデータ
	filename : 拡張子の前までのパス
	*/

	size_t rows = data.rows();
	size_t cols = data.cols();
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> Data;
	Data = data;


	std::ofstream fs1(filename + ".txt");
	fs1 << "rows = " << rows << std::endl;
	fs1 << "cols = " << cols << std::endl;
	fs1 << typeid(data).name() << std::endl;
	fs1.close();

	std::vector<T> save_data(rows * cols);
	Data.resize(data.rows()*data.cols(), 1);
	for (size_t i = 0; i < save_data.size(); i++)
		save_data[i] = Data(i, 0);
	write_vector(save_data, filename + ".raw");
	Data.resize(rows, cols);
}

/*
* @auther tozawa
* @history
* 20171127
* write_raw_and_txtを列優先でrawに保存するバージョン
*/

template<typename T>
void write_raw_and_txt_ColMajor(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data, std::string filename)
{
	/*
	eigenのmatrixをraw画像として保存
	列優先で保存
	出力はraw画像とサイズと型が記されたtxtファイル
	data : 保存するデータ
	filename : 拡張子の前までのパス
	*/
	size_t rows = data.rows();
	size_t cols = data.cols();

	std::ofstream fs1(filename + "_colMajor.txt");
	fs1 << "rows = " << rows << std::endl;
	fs1 << "cols = " << cols << std::endl;
	fs1 << typeid(data).name() << std::endl;
	fs1.close();

	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Data;
	Data = data;
	std::vector<T> save_data(rows * cols);
	Data.resize(data.rows()*data.cols(), 1);
	for (size_t i = 0; i < save_data.size(); i++)
	{
		save_data[i] = Data(i, 0);
	}
	// 美しいが，EigenMatrixが一列のアドレスを確保しているか確かめていないので危険と判断
	//std::vector<T> save_data(data.data(), data.data() + data.rows()*data.cols());
	write_vector(save_data, filename + "_colMajor.raw");
	Data.resize(rows, cols); //もとに戻す？
}

/*
* @auther tozawa
* @history
* 20171127
* load_raw_to_eigenを列優先で保存されたrawをEigenに読み込むバージョン
*/
template<typename T>
void load_raw_to_eigen_ColMajor(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& eigen, std::string filename, int row_size)
{
	std::vector<T> v;
	auto num = get_file_size(filename) / sizeof(T);
	FILE *fp;
	if (fopen_s(&fp, filename.c_str(), "rb") != 0) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	v.resize(num);
	fread(v.data(), sizeof(T), num, fp);

	fclose(fp);

	size_t col_size = v.size() / row_size;

	eigen = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(&v[0], row_size, col_size);
}
