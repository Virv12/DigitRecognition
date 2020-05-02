#include <vector>

struct Layer {
	virtual std::vector<float> operator() (std::vector<float>&) = 0;
	virtual std::vector<float> backprop(std::vector<float>&, std::vector<float>&, const std::vector<float>&) = 0;
	virtual void apply() {}
};

struct LayerLinear : Layer {
	size_t I, O;
	float *W, *A;

	LayerLinear(size_t I, size_t O);

	virtual std::vector<float> operator() (std::vector<float>&) override;
	virtual std::vector<float> backprop(std::vector<float>&, std::vector<float>&, const std::vector<float>&) override;
	virtual void apply() override;
};

struct LayerSigmoid : Layer {
	virtual std::vector<float> operator() (std::vector<float>& m) override;
	virtual std::vector<float> backprop(std::vector<float>& m, std::vector<float>& c, const std::vector<float>& p) override;
};

struct NN {
	std::vector<Layer*> layers;

	NN(std::initializer_list<Layer*> il);
	std::vector<float> operator() (std::vector<float> I);
	void backprop(std::vector<float> I, const std::vector<float>& O);
	void apply();
};
