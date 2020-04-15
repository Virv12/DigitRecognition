CC=g++ -std=c++17 -g -Og -Wall -Wextra -Iincludes -D_GLIBCXX_DEBUG -fsanitize=address
CC=g++ -std=c++17 -Ofast -Wall -Wextra -Iincludes

k-NN: $(addprefix obj/,k-NN.o dataset.o)
	$(CC) -o $@ $^

obj/%.o: src/%.cpp | obj/
	$(CC) -c -o $@ $^

%/:
	mkdir -p $*

clean:
	rm -rf obj k-NN

.PHONY: run clean
.SECONDARY:
