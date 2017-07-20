
INPUT= $(wildcard *.mgm)
INPUT= $(filter-out init.mgm, $(INPUT))
OUTPUT=	$(subst .mgm,.dat,$(INPUT))

%.dat:	%.mgm
	magma $<

all:	$(OUTPUT)

clean:
	rm -f *dat
