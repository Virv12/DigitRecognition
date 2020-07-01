#include <iostream>
#include <fstream>
#include <array>
#include <vector>

#include <dataset.h>
using namespace std;

vector<uint8_t> train_labels;
vector<array<uint8_t, 28*28>> train_images;
vector<uint8_t> test_labels;
vector<array<uint8_t, 28*28>> test_images;

void load_dataset() {
	ifstream fin;
	uint32_t MN, N, A, B;

	/*---------------------*\
	\*---------------------*/

	fin = ifstream("dataset/train-labels-idx1-ubyte");
	fin.read((char*)&MN, sizeof(MN));
	fin.read((char*)&N, sizeof(N));
	N = __builtin_bswap32(N);
	
	if (MN != 0x01080000) {
		cerr << "Invalid database" << endl;
		exit(1);
	}

	train_labels.resize(N);
	fin.read((char*)train_labels.data(), N * sizeof(uint8_t));

	/*---------------------*\
	\*---------------------*/

	fin = ifstream("dataset/train-images-idx3-ubyte");
	fin.read((char*)&MN, sizeof(MN));
	fin.read((char*)&N, sizeof(N));
	fin.read((char*)&A, sizeof(A));
	fin.read((char*)&B, sizeof(B));

	N = __builtin_bswap32(N);
	A = __builtin_bswap32(A);
	B = __builtin_bswap32(B);

	if (MN != 0x03080000 || A != 28 || B != 28) {
		cerr << "invalid database" << endl;
		exit(1);
	}

	train_images.resize(N);
	fin.read((char*)train_images.data(), N * sizeof(array<uint8_t, 28*28>));

	/*---------------------*\
	\*---------------------*/

	fin = ifstream("dataset/t10k-labels-idx1-ubyte");
	fin.read((char*)&MN, sizeof(MN));
	fin.read((char*)&N, sizeof(N));
	N = __builtin_bswap32(N);

	if (MN != 0x01080000) {
		cerr << "Invalid database" << endl;
		exit(1);
	}

	test_labels.resize(N);
	fin.read((char*)test_labels.data(), N * sizeof(uint8_t));

	/*---------------------*\
	\*---------------------*/

	fin = ifstream("dataset/t10k-images-idx3-ubyte");
	fin.read((char*)&MN, sizeof(MN));
	fin.read((char*)&N, sizeof(N));
	fin.read((char*)&A, sizeof(A));
	fin.read((char*)&B, sizeof(B));

	N = __builtin_bswap32(N);
	A = __builtin_bswap32(A);
	B = __builtin_bswap32(B);

	if (MN != 0x03080000 || A != 28 || B != 28) {
		cerr << "invalid database" << endl;
		exit(1);
	}

	test_images.resize(N);
	fin.read((char*)test_images.data(), N * sizeof(array<uint8_t, 28*28>));
}
