#include <array>
#include <vector>

struct Layer {
	virtual ~Layer() = default;

	virtual std::vector<float> operator() (std::vector<float>&) = 0;
	virtual std::vector<float> backprop(std::vector<float>&, std::vector<float>&, const std::vector<float>&) = 0;
	virtual void apply() {}

	virtual void save(std::ofstream&) = 0;

	static Layer* fromFile(int idx, std::ifstream& fin);
};

struct LayerLinear : Layer {
	size_t I, O;
	float *W, *A;

	LayerLinear(size_t I, size_t O);
	~LayerLinear() override;

	LayerLinear(std::ifstream&);
	virtual void save(std::ofstream&) override;

	virtual std::vector<float> operator() (std::vector<float>&) override;
	virtual std::vector<float> backprop(std::vector<float>&, std::vector<float>&, const std::vector<float>&) override;
	virtual void apply() override;
};

struct LayerSigmoid : Layer {
	virtual void save(std::ofstream&) override;

	virtual std::vector<float> operator() (std::vector<float>& m) override;
	virtual std::vector<float> backprop(std::vector<float>& m, std::vector<float>& c, const std::vector<float>& p) override;
};

struct LayerAveragePooling : Layer {
	std::array<size_t, 2> D, S;

	LayerAveragePooling(std::array<size_t, 2> S, std::array<size_t, 2> D) : D(D), S(S) {}

	LayerAveragePooling(std::ifstream&);
	virtual void save(std::ofstream&) override;

	virtual std::vector<float> operator() (std::vector<float>&) override;
	virtual std::vector<float> backprop(std::vector<float>&, std::vector<float>&, const std::vector<float>&) override;
};

struct LayerConvolutional : Layer {
	size_t I, O;
	std::array<size_t, 2> S, K;
	float *W, *A;

	LayerConvolutional(size_t, size_t, std::array<size_t, 2>, std::array<size_t, 2>);
	~LayerConvolutional() override;

	LayerConvolutional(std::ifstream&);
	virtual void save(std::ofstream&) override;

	virtual std::vector<float> operator() (std::vector<float>&) override;
	virtual std::vector<float> backprop(std::vector<float>&, std::vector<float>&, const std::vector<float>&) override;
	virtual void apply() override;
};

struct NN {
	std::vector<Layer*> layers;

	NN(std::initializer_list<Layer*> il);
	~NN();

	NN(std::string path);
	void save(std::string);

	std::vector<float> operator() (std::vector<float> I);
	void backprop(std::vector<float> I, const std::vector<float>& O);
	void apply();
};
