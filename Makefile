# CC=clang++ -std=c++20 -g -Ofast -Wall -Wextra -Iincludes -D_GLIBCXX_DEBUG -fsanitize=address
CC=clang++ -std=c++20 -Ofast -Wall -Wextra -Wpedantic -Iincludes -DNDEBUG

DATASET=$(addprefix dataset/,t10k-labels-idx1-ubyte train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte)
BINARY=$(addprefix bin/,k-NN dnn tnn)

all: $(BINARY) $(DATASET)

bin/k-NN: $(addprefix obj/,k-NN.o dataset.o)
bin/dnn: $(addprefix obj/,dnn.o nn.o dataset.o)
bin/tnn: $(addprefix obj/,tnn.o nn.o dataset.o)

$(BINARY): bin/% : | bin/
	$(CC) -o $@ $^

obj/%.o: src/%.cpp | obj/
	$(CC) -c -o $@ $^

$(DATASET): dataset/% : | dataset/
	curl 'http://yann.lecun.com/exdb/mnist/$*.gz' -o $@.gz 2>/dev/null
	gunzip $@.gz

%/:
	mkdir -p $*

clean:
	rm -rf obj dataset bin

.PHONY: run clean
.SECONDARY:
