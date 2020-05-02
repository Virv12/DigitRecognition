# Digit Recognition
Handwritten digit recognition implemented in c++ without libraries

# Current algorithms
- [x] k-NN
- [ ] k-NN with [k-d tree](https://en.wikipedia.org/wiki/K-d_tree)
- [x] Deep Neural Network - Backpropagation
- [ ] Convolutional Neural Network

# Results
## k-NN
Best result with 3 closest neighbours from a 60000 images dataset.  
Error rate: 2.95%

# dnn
Error rate: 3.87%

# Compilation
```bash
make -j4
```

Dataset: <http://yann.lecun.com/exdb/mnist/>
