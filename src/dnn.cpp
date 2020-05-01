#include <assert.h>
#include <math.h>

#include <numeric>
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
		data = new float[N * M];
	}

	~matrix() {
		delete[] data;
	}

	inline float* operator[] (const size_t i) const {
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

			for (size_t j = 0; j < W.back().N * W.back().M; j++) {
				W.back().data[j] = (float)rand() / RAND_MAX * 2 - 1;
				A.back().data[j] = 0;
			}
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

	void backprop(const vector<float>& I, const vector<float>& O) {
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
					W[i][j][k] -= 0.01 * A[i][j][k];
					A[i][j][k] = 0;
				}
	}
};

int main() {
	srand(time(0));

	load_dataset();
	NN nn({28*28, 28, 28, 10});

	vector<int> S(train_labels.size());
	iota(S.begin(), S.end(), 0);

	do {
		random_shuffle(S.begin(), S.end());
		for (size_t i = 0; i < S.size(); i++) {
			vector<float> I(train_images[S[i]].begin(), train_images[S[i]].end());
			for (float& f : I)
				f /= 255;

			vector<float> O(10, 0);
			O[train_labels[S[i]]] = 1;

			nn.backprop(I, O);

			if ((i + 1) % 10 == 0)
				nn.apply();
		}

		size_t C = 0;
		for (size_t i = 0; i < test_labels.size(); i++) {
			vector<float> I(test_images[i].begin(), test_images[i].end());
			for (float& f : I)
				f /= 255;

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
