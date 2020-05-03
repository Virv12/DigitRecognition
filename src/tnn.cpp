#include <cstdlib>
#include <ctime>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <dataset.h>
#include <nn.h>
using namespace std;

int main() {
	srand(time(0));
	load_dataset();

	NN nn("nn.bin");

	size_t C = 0;
	for (size_t i = 0; i < test_labels.size(); i++) {
		vector<float> I(28*28);
		for (size_t j = 0; j < I.size(); j++)
			I[j] = test_images[i][j] / 255.0f;

		vector<float> O = nn(I);

		C += max_element(O.begin(), O.end()) - O.begin() == test_labels[i];
	}

	float P = 100.0f * C / test_labels.size();
	printf("[");
	for (size_t i = 0; i < 100; i++)
		printf("%c", i < (P - 90) * 10 ? '#' : ' ');
	printf("] %.2f%%\n", P);
}
