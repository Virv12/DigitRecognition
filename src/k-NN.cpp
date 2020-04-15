#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include "dataset.h"
using namespace std;

size_t K = 3;
/* 0:   09.80%
 * 1:   96.91%
 * 2:   96.27%
 * 3:   97.05%
 * 4:   96.82%
 * 5:   96.88%
 * 6:   96.77%
 * 7:   96.94%
 * 8:   96.70%
 * 9:   96.59%
 * 10:  96.65%
 * 20:  96.25%
 * 30:  95.96%
 * 40:  95.60%
 * 100: 94.40%
 * 60000: 11.35%
 */

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

int main(int argc, char** argv) {
	if (1 < argc)
		K = stoi(argv[1]);

	load_dataset();

	size_t C = 0;
	for (size_t i = 0; i < test_labels.size(); i++) {
		C += test_labels[i] == predict(i);
		if (i % 20 == 19)
			cerr << C << " / " << i + 1 << '\n';
	}
	cout << K << ": " << C << " / " << test_labels.size() << endl;
}
