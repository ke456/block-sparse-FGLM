CXX=g++
CXXFLAGS= -I/usr/local/include -O2 -Wall -g -std=gnu++11 -fabi-version=6 -fopenmp -std=gnu++11 -mmmx -mpopcnt -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mbmi -mbmi2 -mfpmath=sse -fabi-version=6 -I/usr/local/include -std=gnu++11 
LDFLAGS= -L/usr/local/lib -llinbox -L/usr/local/lib -fopenmp -lopenblas -lgivaro -lgmp -lgmpxx -lntl      

HDR= $(wildcard *.h)
SRC= $(wildcard *.cc)
OBJ= $(subst .cc,.o,$(SRC))
TGT= main

all: $(TGT)

%.o: %.cc $(HDR)
	$(CXX) $(CXXFLAGS) -c $<

main: $(SRC) $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)

clean:
	rm $(OBJ) 

cleanall:
	rm $(OBJ) $(TGT)

