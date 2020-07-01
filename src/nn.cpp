#include <cassert>
#include <cmath>
#include <vector>
#include <fstream>

#include <nn.h>
using namespace std;

Layer* Layer::fromFile(int idx, ifstream& fin) {
	switch (idx) {
		case 0: return new LayerLinear(fin);
		case 1: return new LayerSigmoid;
		case 2: return new LayerAveragePooling(fin);
		case 3: return new LayerConvolutional(fin);
		default: assert(false); return nullptr;
	}
}

LayerLinear::LayerLinear(size_t I, size_t O) : I(I), O(O) {
	size_t T = (I + 1) * O;
	W = new float[T];
	A = new float[T] {};
	for (size_t i = 0; i < T; i++)
		W[i] = (float)rand() / RAND_MAX * 2 - 1;
}

LayerLinear::~LayerLinear() {
	delete[] W;
	delete[] A;
}

LayerLinear::LayerLinear(ifstream& fin) {
	fin.read((char*)&I, sizeof(I));
	fin.read((char*)&O, sizeof(O));

	size_t T = O * (I + 1);
	W = new float[T];
	A = new float[T] {};
	fin.read((char*)W, T * sizeof(*W));
}

void LayerLinear::save(ofstream& fout) {
	fout.write("\000", 1);
	fout.write((char*)&I, sizeof(I));
	fout.write((char*)&O, sizeof(O));
	fout.write((char*)W, O * (I + 1) * sizeof(*W));
}

vector<float> LayerLinear::operator() (vector<float>& m) {
	assert(m.size() == I);

	vector<float> r(O);
	for (size_t o = 0; o < O; o++) {
		r[o] = W[o * (I + 1) + I];
		for (size_t i = 0; i < I; i++)
			r[o] += m[i] * W[o * (I + 1) + i];
	}
	return r;
}

vector<float> LayerLinear::backprop(vector<float>& m, vector<float>&, const vector<float>& p) {
	assert(m.size() == O);
	// assert(c.size() == O);
	assert(p.size() == I);

	vector<float> r(I);
	for (size_t o = 0; o < O; o++) {
		A[o * (I + 1) + I] += m[o];
		for (size_t i = 0; i < I; i++) {
			A[o * (I + 1) + i] += p[i] * m[o];
			r[i] += W[o * (I + 1) + i] * m[o];
		}
	}
	return r;
}

void LayerLinear::apply() {
	for (size_t i = 0; i < O * (I + 1); i++) {
		W[i] -= 0.03 * A[i];
		A[i] = 0;
	}
}

void LayerSigmoid::save(ofstream& fout) {
	fout.write("\001", 1);
}

vector<float> LayerSigmoid::operator() (vector<float>& m) {
	for (size_t i = 0; i < m.size(); i++)
		m[i] = 1 / (1 + exp(-m[i]));
	return m;
}

vector<float> LayerSigmoid::backprop(vector<float>& m, vector<float>& c, const vector<float>&) {
	// assert(m.size() == p.size());
	assert(m.size() == c.size());

	for (size_t i = 0; i < m.size(); i++)
		m[i] *= c[i] * (1 - c[i]);
	return m;
}

LayerAveragePooling::LayerAveragePooling(ifstream& fin) {
	fin.read((char*)&D, sizeof(D));
	fin.read((char*)&S, sizeof(S));
}

void LayerAveragePooling::save(ofstream& fout) {
	fout.write("\002", 1);
	fout.write((char*)&D, sizeof(D));
	fout.write((char*)&S, sizeof(S));
}

vector<float> LayerAveragePooling::operator() (vector<float>& m) {
	assert(m.size() % (S[0] * D[0] * S[1] * D[1]) == 0);

	size_t I = m.size() / (S[0] * D[0] * S[1] * D[1]);
	vector<float> r(I * S[0] * S[1]);

	for (size_t i = 0; i < I; i++)
		for (size_t x = 0; x < S[0] * D[0]; x++)
			for (size_t y = 0; y < S[1] * D[1]; y++)
				r[(i * S[0] + x / D[0]) * S[1] + y / D[1]] += m[(i * S[0] * D[0] + x) * S[1] * D[1] + y] / (D[0] * D[1]);

	return r;
}

vector<float> LayerAveragePooling::backprop(vector<float>& m, vector<float>&, const vector<float>&) {
	assert(m.size() % (S[0] * S[1]) == 0);

	size_t I = m.size() / (S[0] * S[1]);
	vector<float> r(I * S[0] * D[0] * S[1] * D[1]);

	for (size_t i = 0; i < I; i++)
		for (size_t x = 0; x < S[0] * D[0]; x++)
			for (size_t y = 0; y < S[1] * D[1]; y++)
				r[(i * S[0] * D[0] + x) * S[1] * D[1] + y] = m[(i * S[0] + x / D[0]) * S[1] + y / D[1]] / (D[0] * D[1]);

	return r;
}

LayerConvolutional::LayerConvolutional(size_t I, size_t O, array<size_t, 2> S, array<size_t, 2> K)
	: I(I), O(O), S(S), K(K) {
		size_t T = O * (I * K[0] * K[1] + 1);
		W = new float[T];
		A = new float[T] {};
		for (size_t i = 0; i < T; i++)
			W[i] = (float)rand() / RAND_MAX * 2 - 1;
}

LayerConvolutional::~LayerConvolutional() {
	delete[] W;
	delete[] A;
}

LayerConvolutional::LayerConvolutional(ifstream& fin) {
	fin.read((char*)&I, sizeof(I));
	fin.read((char*)&O, sizeof(O));
	fin.read((char*)&S, sizeof(S));
	fin.read((char*)&K, sizeof(K));

	size_t T = O * (I * K[0] * K[1] + 1);
	W = new float[T];
	A = new float[T] {};
	fin.read((char*)W, T * sizeof(*W));
}

void LayerConvolutional::save(ofstream& fout) {
	fout.write("\003", 1);
	fout.write((char*)&I, sizeof(I));
	fout.write((char*)&O, sizeof(O));
	fout.write((char*)&S, sizeof(S));
	fout.write((char*)&K, sizeof(K));
	fout.write((char*)W, O * (I * K[0] * K[1] + 1) * sizeof(*W));
}

vector<float> LayerConvolutional::operator() (vector<float>& m) {
	assert(m.size() == I * (S[0] + K[0] - 1) * (S[1] + K[1] - 1));

	vector<float> r(O * S[0] * S[1]);

	for (size_t o = 0; o < O; o++)
		for (size_t x = 0; x < S[0]; x++)
			for (size_t y = 0; y < S[1]; y++) {
				r[(o * S[0] + x) * S[1] + y] = W[(o + 1) * (I * K[0] * K[1] + 1) - 1];

				for (size_t i = 0; i < I; i++)
					for (size_t dx = 0; dx < K[0]; dx++)
						for (size_t dy = 0; dy < K[1]; dy++)
							r[(o * S[0] + x) * S[1] + y] +=
								m[(i * (S[0] + K[0] - 1) + x + dx) * (S[1] + K[1] - 1) + y + dy] *
								W[o * (I * K[0] * K[1] + 1) + (i * K[0] + dx) * K[1] + dy];
			}

	return r;
}

vector<float> LayerConvolutional::backprop(vector<float>& m, vector<float>&, const vector<float>& p) {
	assert(m.size() == O * S[0] * S[1]);
	// assert(c.size() == O * S[0] * S[1]);
	assert(p.size() == I * (S[0] + K[0] - 1) * (S[1] + K[1] - 1));

	vector<float> r(I * (S[0] + K[0] - 1) * (S[1] + K[1] - 1));

	for (size_t o = 0; o < O; o++)
		for (size_t x = 0; x < S[0]; x++)
			for (size_t y = 0; y < S[1]; y++) {
				A[(o + 1) * (I * K[0] * K[1] + 1) - 1] += m[(o * S[0] + x) * S[1] + y];

				for (size_t i = 0; i < I; i++)
					for (size_t dx = 0; dx < K[0]; dx++)
						for (size_t dy = 0; dy < K[1]; dy++) {
							A[o * (I * K[0] * K[1] + 1) + (i * K[0] + dx) * K[1] + dy] +=
								p[(i * (S[0] + K[0] - 1) + x + dx) * (S[1] + K[1] - 1) + y + dy] *
								m[(o * S[0] + x) * S[1] + y];

							r[(i * (S[0] + K[0] - 1) + x + dx) * (S[1] + K[1] - 1) + y + dy] +=
								W[o * (I * K[0] * K[1] + 1) + (i * K[0] + dx) * K[1] + dy] *
								m[(o * S[0] + x) * S[1] + y];
						}
			}

	return r;
}

void LayerConvolutional::apply() {
	for (size_t i = 0; i < O * I * K[0] * K[1]; i++) {
		W[i] -= 0.03 * A[i];
		A[i] = 0;
	}
}

NN::NN(initializer_list<Layer*> il) : layers(il) {}

NN::~NN() {
	for (Layer* l : layers)
		delete l;
}

vector<float> NN::operator() (vector<float> I) {
	for (Layer* l : layers) {
		assert(l);
		I = (*l)(I);
	}
	return I;
}

void NN::backprop(vector<float> I, const vector<float>& O) {
	vector<vector<float>> neurons { I };

	for (Layer* l : layers) {
		I = (*l)(I);
		neurons.push_back(I);
	}

	assert(neurons.back().size() == O.size());

	vector<float> D(neurons.back().size());
	for (size_t i = 0; i < D.size(); i++)
		D[i] = 2 * (neurons.back()[i] - O[i]);

	for (size_t i = layers.size() - 1; i < layers.size(); i--)
		D = layers[i]->backprop(D, neurons[i + 1], neurons[i]);
}

void NN::apply() {
	for (Layer* l : layers)
		l->apply();
}

void NN::save(string path) {
	ofstream fout(path);
	for (Layer* l : layers)
		l->save(fout);
	fout.close();
}

NN::NN(string path) {
	ifstream fin(path);
	char c;
	while (c = fin.get(), !fin.eof()) {
		Layer* l = Layer::fromFile(c, fin);
		assert(l);
		layers.push_back(l);
	}
	fin.close();
}
