CXX=g++
CXXFLAGS= -I/usr/local/include -O2 -Wall -g -fopenmp -fabi-version=6 -mmmx -mpopcnt -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mbmi -mbmi2 -mfpmath=sse -std=gnu++11 -Wno-sign-compare
LDFLAGS= -L/usr/local/lib -fopenmp -lopenblas -lgivaro -lgmp -lgmpxx -lntl -lmpfr -llinbox 

HDR:= $(wildcard *.h)
SRC:= $(wildcard *.cc)
TST:= $(wildcard test*.cc)
SRC:= $(filter-out $(TST), $(SRC))
OBJ:= $(subst .cc,.o,$(SRC))
TGT:= fglm polmatdom

all: $(TGT)

%.o: %.cc $(HDR)
	$(CXX) $(CXXFLAGS) -c $< $(DFLAGS)

fglm: $(SRC) $(TST) $(OBJ)
	$(CXX) -o fglm test_fglm.cc $(OBJ) $(LDFLAGS) $(DFLAGS)

polmatdom: $(SRC) $(OBJ)
	$(CXX) -DTEST_POLMATDOM -o polmatdom test_polmatdom.cc $(OBJ) $(LDFLAGS) $(DFLAGS)

clean:
	rm $(OBJ) 

cleanall:
	rm $(OBJ) $(TGT)

