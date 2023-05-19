.PHONY: all
all: dgemm dgemm_unroll dgemm_blocking

.PHONY: prepare
prepare:
	mkdir -p ./out
	mkdir -p ./out_csv/raw

.PHONY: dgemm
dgemm: prepare
	gcc  -O3 -fopenmp -fopenmp -march=native -o out/dgemm src/main.c src/dgemm.c -lm

.PHONY: csv_all
csv_all:

.PHONY: csv_1024
csv_1024: 
	./scripts/create_csv.py -u 8 -b 32 -o 1024 -f dgemm.1024.csv

.PHONY: csv
csv_2048: 
	./scripts/create_csv.py -u 8 -b 32 -o 2048 -f dgemm.2048.csv

.PHONY: csv
csv_4096: 
	./scripts/create_csv.py -u 8 -b 32 -o 4096 -f dgemm.4096.csv

.PHONY: clean
clean:
	rm -fr ./out/*
	rm -fr ./out_csv/*
	rm -fr ./out_csv/raw/*
