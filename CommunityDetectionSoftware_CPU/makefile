SRC=main.cpp Community.cpp ./SVDLIBC/*s*.c
CFLAGS=-L./SVDLIBC -lsvd -lm
COPTZ=-O3 -g #-fopenmp
SUPP_WARN=-Wno-write-strings

build: # main.cpp Community.cpp Community.h
	g++ -o EXEC $(SRC) $(CFLAGS) $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

run:
	./EXEC

clean:
	rm EXEC
