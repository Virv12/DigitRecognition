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

	// NN nn {
	// 	new LayerLinear(28*28, 28),
	// 	new LayerSigmoid,
	// 	new LayerLinear(28   , 28),
	// 	new LayerSigmoid,
	// 	new LayerLinear(28   , 10),
	// 	new LayerSigmoid,
	// };
	
	NN nn {
		new LayerAveragePooling({14, 14}, {2, 2}),
		new LayerConvolutional(1, 2, {10, 10}, {5, 5}),
		new LayerSigmoid,

		new LayerLinear(2*10*10, 10),
		new LayerSigmoid,
	};

	vector<int> S(train_labels.size());
	iota(S.begin(), S.end(), 0);

	do {
		random_shuffle(S.begin(), S.end());
		for (size_t i = 0; i < S.size(); i++) {
			vector<float> I(28*28);
			for (size_t j = 0; j < I.size(); j++)
				I[j] = train_images[S[i]][j] / 255.0f;

			vector<float> O(10);
			O[train_labels[S[i]]] = 1;

			nn.backprop(I, O);

			if ((i + 1) % 10 == 0)
				nn.apply();

			if ((i + 1) % 500 == 0) {
				printf("%lu / %lu\r", i+1, S.size());
				fflush(stdout);
			}
		}

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
	} while (true);
}
