#include <assert.h>
#include <math.h>

#include <algorithm>
#include <array>
#include <vector>

#include <dataset.h>
using namespace std;

float sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

struct matrix {
	float* data;
	size_t N, M;

	matrix(size_t N, size_t M) : N(N), M(M) {
		data = new float[N * M] {};
	}

	~matrix() {
		delete[] data;
	}

	inline float* operator[] (size_t i) {
		assert(i < N);
		return data + i * M;
	}
};

struct NN {
	vector<matrix> W;
	vector<matrix> A;

	NN(vector<int> D) {
		W.reserve(D.size() - 1);
		A.reserve(D.size() - 1);

		for (size_t i = 1; i < D.size(); i++) {
			W.emplace_back(D[i], D[i-1] + 1);
			A.emplace_back(D[i], D[i-1] + 1);
		}
	}

	vector<float> operator() (vector<float> I) {
		for (matrix& m : W) {
			assert(I.size() + 1 == m.M);
			vector<float> O(m.N);

			for (size_t i = 0; i < O.size(); i++) {
				for (size_t j = 0; j < I.size(); j++)
					O[i] += I[j] * m[i][j];
				O[i] = sigmoid(O[i] + m[i][I.size()]);
			}

			I = move(O);
		}

		return I;
	}

	void backprop(vector<float> I, vector<float> O) {
		vector<vector<float>> NN { I };

		for (matrix& m : W) {
			vector<float> I = NN.back();
			vector<float>& O = NN.emplace_back(m.N);

			assert(m.M);
			assert(I.size() + 1);
			assert(I.size() + 1 == m.M);

			for (size_t i = 0; i < O.size(); i++) {
				for (size_t j = 0; j < I.size(); j++)
					O[i] += I[j] * m[i][j];
				O[i] = sigmoid(O[i] + m[i][I.size()]);
			}
		}

		vector<float> D(NN.back().size());
		assert(D.size() == O.size());

		for (size_t i = 0; i < D.size(); i++)
			D[i] = 2 * (NN.back()[i] - O[i]);

		for (size_t i = NN.size() - 2; i < NN.size(); i--) {
			for (size_t j = 0; j < D.size(); j++)
				D[j] *= NN[i + 1][j] * (1 - NN[i + 1][j]);

			vector<float> T(W[i].M - 1);
			for (size_t j = 0; j < D.size(); j++) {
				for (size_t k = 0; k < T.size(); k++) {
					T[k] += W[i][j][k] * D[j];
					A[i][j][k] += NN[i][k] * D[j];
				}
				A[i][j][T.size()] += D[j];
			}

			D = move(T);
		}
	}

	void apply() {
		for (size_t i = 0; i < W.size(); i++)
			for (size_t j = 0; j < W[i].N; j++)
				for (size_t k = 0; k < W[i].M; k++) {
					W[i][j][k] -= 0.1 * A[i][j][k];
					A[i][j][k] = 0;
				}
	}
};

int main() {
	load_dataset();
	NN nn({28*28, 16, 16, 10});

	do {
		for (size_t i = 0; i < train_labels.size(); i++) {
			vector<float> I(train_images[i].begin(), train_images[i].end());
			for (float& f : I)
				f /= 255;

			vector<float> O(10, 0);
			O[train_labels[i]] = 1;

			nn.backprop(I, O);

			if ((i + 1) % 100 == 0)
				nn.apply();

			if ((i + 1) % 2000 == 0) {
				printf("%lu / %lu\r", i, train_labels.size());
				fflush(stdout);
			}
		}

		size_t C = 0;
		for (size_t i = 0; i < test_labels.size(); i++) {
			vector<float> I(test_images[i].begin(), test_images[i].end());
			for (float& f : I)
				f /= 255;

			vector<float> O = nn(I);

			C += max_element(O.begin(), O.end()) - O.begin() == test_labels[i];
		}
		printf("%lu / %lu: %f%%\n", C, test_labels.size(), 100.0f * C / test_labels.size());
	} while (true);
}
