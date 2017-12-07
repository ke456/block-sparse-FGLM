CXX=g++
CXXFLAGS= -I/usr/local/include -I/usr/local/include/eigen3 -O2 -Wall -g -fopenmp -fabi-version=6 -mmmx -mpopcnt -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mbmi -mbmi2 -mfpmath=sse -std=gnu++11 -Wno-sign-compare -DEIGEN_DONT_PARALLELIZE
LDFLAGS= -L/usr/local/lib -fopenmp -lopenblas -lgivaro -lgmp -lgmpxx -lntl -lmpfr -llinbox 

HDR:= $(wildcard *.h)
SRCS:= $(wildcard *sparse*.cc)
SRCD:= $(wildcard *dense*.cc)
TST:= $(wildcard test*.cc)
SRCS:= $(filter-out $(TST), $(SRCS))
SRCD:= $(filter-out $(TST), $(SRCD))
OBJS:= $(subst .cc,.o,$(SRCS))
OBJD:= $(subst .cc,.o,$(SRCD))
TGT:= fglm polmatdom

all: $(TGT)

%.o: %.cc $(HDR)
	$(CXX) $(CXXFLAGS) -c $< $(DFLAGS)

sfglm: $(SRCS) $(TST) $(OBJS)
	echo $(SRCS)
	$(CXX) -o sfglm test_fglm.cc $(OBJS) $(LDFLAGS) $(DFLAGS)

dfglm: $(SRCD) $(TST) $(OBJD)
	$(CXX) -o dfglm test_fglm.cc $(OBJD) $(LDFLAGS) $(DFLAGS)

polmatdom: $(SRC) $(OBJ)
	$(CXX) -DTEST_POLMATDOM -o polmatdom test_polmatdom.cc $(OBJ) $(LDFLAGS) $(DFLAGS)

clean:
	rm $(OBJS) $(OBJD) 

cleanall:
	rm $(OBJ) $(TGT)

eigen:	$(TST)
	$(CXX) -o eigen test-eigen.cc  $(CXXFLAGS) $(LDFLAGS) $(DFLAGS)
