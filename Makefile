CC=g++ -std=c++17 -g -Og -Wall -Wextra -Iincludes -D_GLIBCXX_DEBUG -fsanitize=address

obj/%.o: src/%.cpp | obj/
	$(CC) -c -o $@ $^

%/:
	mkdir -p $*

clean:
	rm -rf obj client server

.PHONY: run clean
.SECONDARY:
