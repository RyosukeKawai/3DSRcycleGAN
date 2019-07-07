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
	�t�@�C���̃T�C�Y���擾����v���O����
	filename : �t�@�C����
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
	raw�摜��ǂݍ����vector�Ɋi�[
	v : �i�[����x�N�^�[
	filename : �t�@�C����
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
	vector��raw�摜�ɏ�������
	v : �i�[����x�N�^�[
	filename : �ۑ��ꏊ��΃p�X
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

	// �������ȁ`�Ǝv���ĕύX(20171121 tozawa)
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
	eigen��matrix��raw�摜�Ƃ��ĕۑ�
	�o�͂�raw�摜�ƃT�C�Y�ƌ^���L���ꂽtxt�t�@�C��
	data : �ۑ�����f�[�^
	filename : �g���q�̑O�܂ł̃p�X
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
* write_raw_and_txt���D���raw�ɕۑ�����o�[�W����
*/

template<typename T>
void write_raw_and_txt_ColMajor(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data, std::string filename)
{
	/*
	eigen��matrix��raw�摜�Ƃ��ĕۑ�
	��D��ŕۑ�
	�o�͂�raw�摜�ƃT�C�Y�ƌ^���L���ꂽtxt�t�@�C��
	data : �ۑ�����f�[�^
	filename : �g���q�̑O�܂ł̃p�X
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
	// ���������CEigenMatrix�����̃A�h���X���m�ۂ��Ă��邩�m���߂Ă��Ȃ��̂Ŋ댯�Ɣ��f
	//std::vector<T> save_data(data.data(), data.data() + data.rows()*data.cols());
	write_vector(save_data, filename + "_colMajor.raw");
	Data.resize(rows, cols); //���Ƃɖ߂��H
}

/*
* @auther tozawa
* @history
* 20171127
* load_raw_to_eigen���D��ŕۑ����ꂽraw��Eigen�ɓǂݍ��ރo�[�W����
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
