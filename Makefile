# CC=g++ -std=c++17 -g -Og -Wall -Wextra -Iincludes -D_GLIBCXX_DEBUG -fsanitize=address
CC=g++ -std=c++17 -Ofast -Wall -Wextra -Iincludes
DATASET=$(addprefix dataset/,t10k-labels-idx1-ubyte train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte)
BINARY=$(addprefix bin/,k-NN dnn)

all: $(BINARY) $(DATASET)

bin/k-NN: $(addprefix obj/,k-NN.o dataset.o)
bin/dnn: $(addprefix obj/,dnn.o dataset.o)

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
