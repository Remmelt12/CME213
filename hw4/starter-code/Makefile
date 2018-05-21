CUD=nvcc
CUDFLAGS=-O3 -std=c++11 --expt-extended-lambda -arch=sm_20 

all: create_cipher solve_cipher

create_cipher: create_cipher.cu
	$(CUD) -o $@ $< $(CUDFLAGS)

solve_cipher: solve_cipher.cu
	$(CUD) -o $@ $< $(CUDFLAGS)

clean:
	rm -f *.o *~ *~ create_cipher solve_cipher
	rm -rf *.dSYM
	rm -f *.err *.out
