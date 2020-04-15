# CC=g++ -std=c++17 -g -Og -Wall -Wextra -Iincludes -D_GLIBCXX_DEBUG -fsanitize=address
CC=g++ -std=c++17 -Ofast -Wall -Wextra -Iincludes
DATASET=t10k-labels-idx1-ubyte train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte

all: k-NN $(DATASET)

k-NN: $(addprefix obj/,k-NN.o dataset.o)
	$(CC) -o $@ $^

obj/%.o: src/%.cpp | obj/
	$(CC) -c -o $@ $^

$(addprefix dataset/,$(DATASET)): dataset/% : | dataset/
	curl 'http://yann.lecun.com/exdb/mnist/$*.gz' -o $@.gz
	gunzip $@.gz

%/:
	mkdir -p $*

clean:
	rm -rf obj dataset k-NN

.PHONY: run clean
.SECONDARY:
