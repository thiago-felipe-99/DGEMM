.PHONY: all
all: dgemm dgemm_unroll dgemm_blocking

.PHONY: prepare
prepare:
	mkdir -p ./out
	mkdir -p ./out_csv/raw
	pip install -r ./requirements.txt

.PHONY: dgemm
dgemm: prepare
	gcc  -O3 -fopenmp -fopenmp -march=native -o out/dgemm src/main.c src/dgemm.c -lm

.PHONY: csv_all
csv_all: prepare csv_1024 csv_2048 csv_4096

.PHONY: csv_1024
csv_1024: 
	python -u ./scripts/create_csv.py -u 8 -b 32 -m 32 -M 1024 -s 32 -f ./out/dgemm.1024.csv

.PHONY: csv
csv_2048: 
	python -u ./scripts/create_csv.py -u 8 -b 32 -m 32 -M 2048 -s 32 -f ./out/dgemm.2048.csv

.PHONY: csv
csv_4096: 
	python -u ./scripts/create_csv.py -u 8 -b 32 -m 2048 -M 4096 -s 128 -f ./out/dgemm.4096.csv

.PHONY: clean
clean:
	rm -fr ./out/*
	rm -fr ./out_csv/*
