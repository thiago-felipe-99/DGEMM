.PHONY: prepare
prepare:
	mkdir -p ./out
	mkdir -p ./out_csv/raw
	pip install -r ./requirements.txt

.PHONY: dgemm
dgemm: prepare
	gcc  -O3 -fopenmp -fopenmp -march=native -o out/dgemm src/main.c src/dgemm.c -lm

.PHONY: csv_all
csv_all: csv_1024 csv_2048 csv_4096

.PHONY: csv_1024
csv_1024: prepare
	python3 -u ./scripts/create_csv.py -u 8 -b 32 -m 32 -M 1024 -s 32 -f ./out/dgemm.1024.csv

.PHONY: csv
csv_2048: prepare
	python3 -u ./scripts/create_csv.py -u 8 -b 32 -m 32 -M 2048 -s 32 -f ./out/dgemm.2048.csv

.PHONY: csv
csv_4096: prepare
	python3 -u ./scripts/create_csv.py -u 8 -b 32 -m 2048 -M 4096 -s 128 -P true -f ./out/dgemm.4096.csv

.PHONY: csv
csv_8192: prepare 
	python3 -u ./scripts/create_csv.py -u 8 -b 32 -m 4096 -M 8192 -s 128 -P true -f ./out/dgemm.8192.csv

.PHONY: figs
figs: prepare
	python3 -u ./scripts/figs_csv.py graficos/notebook/1024.csv 1500 1500 1500 200
	python3 -u ./scripts/figs_csv.py graficos/i5/1024.csv 2000 2000 2000 300
	python3 -u ./scripts/figs_csv.py graficos/i7/1024.csv 1500 1500 1500 200
	python3 -u ./scripts/figs_csv.py graficos/notebook/2048.csv 25000 25000 25000 1500
	python3 -u ./scripts/figs_csv.py graficos/i5/2048.csv 45000 45000 45000 2000
	python3 -u ./scripts/figs_csv.py graficos/i7/2048.csv 15000 15000 15000 1500

.PHONY: clean
clean:
	rm -fr ./out/*
	rm -fr ./out_csv/*
