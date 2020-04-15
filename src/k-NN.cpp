#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include "dataset.h"
using namespace std;

constexpr size_t K = 100; // 94.40%

uint32_t distSQ(size_t i, size_t j) {
	uint32_t d = 0, z;
	for (size_t k = 0; k < 28*28; k++) {
		z = (uint32_t)test_images[i][k] - train_images[j][k];
		d += z * z;
	}
	return d;
}

uint8_t predict(size_t i) {
	vector<pair<uint32_t, uint8_t>> D(train_labels.size());

	for (size_t j = 0; j < train_labels.size(); j++)
		D[j] = { distSQ(i, j), train_labels[j] };

	sort(D.begin(), D.end());

	int F[10] {};
	for (size_t i = 0; i < min(K, D.size()); i++)
		F[D[i].second] ++;

	return max_element(F, F+10) - F;
}

int main() {
	load_dataset();

	size_t C = 0;
	for (size_t i = 0; i < test_labels.size(); i++) {
		C += test_labels[i] == predict(i);
		cout << C << " / " << i + 1 << endl;
	}
}
