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
TGT:= sfglm

all: $(TGT)

%.o: %.cc $(HDR)
	$(CXX) $(CXXFLAGS) -c $< $(DFLAGS)

sfglm: $(SRCS) $(TST) $(OBJS)
	$(CXX) -o sfglm $(OBJS) $(LDFLAGS) $(DFLAGS)

dfglm: $(SRCD) $(TST) $(OBJD)
	$(CXX) -o dfglm $(OBJD) $(LDFLAGS) $(DFLAGS)

clean:
	rm $(OBJS) $(OBJD) 

cleanall:
	rm $(OBJ) $(TGT)
