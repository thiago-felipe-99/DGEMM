.PHONY: all
all: dgemm dgemm_unroll dgemm_blocking dgemm_avx256_unroll dgemm_avx256_blocking dgemm_avx512_unroll dgemm_avx512_blocking

.PHONY: prepare
prepare:
	mkdir -p ./out
	mkdir -p ./out_csv/raw

.PHONY: dgemm
dgemm: prepare
	gcc -O3           -o out/dgemm        src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2    -o out/dgemm_avx256 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -o out/dgemm_avx512 src/main.c src/dgemm.c -lm

.PHONY: dgemm_unroll
dgemm_unroll: 
	gcc -O3 -DUNROLL=2  -o out/dgemm_unroll_02 src/main.c src/dgemm.c -lm
	gcc -O3 -DUNROLL=4  -o out/dgemm_unroll_04 src/main.c src/dgemm.c -lm
	gcc -O3 -DUNROLL=8  -o out/dgemm_unroll_08 src/main.c src/dgemm.c -lm
	gcc -O3 -DUNROLL=16 -o out/dgemm_unroll_16 src/main.c src/dgemm.c -lm
	gcc -O3 -DUNROLL=32 -o out/dgemm_unroll_32 src/main.c src/dgemm.c -lm

.PHONY: dgemm_blocking
dgemm_blocking: 
	gcc -O3 -DBLOCK_SIZE=32  -o out/dgemm_blocking_032 src/main.c src/dgemm.c -lm
	gcc -O3 -DBLOCK_SIZE=64  -o out/dgemm_blocking_064 src/main.c src/dgemm.c -lm
	gcc -O3 -DBLOCK_SIZE=128 -o out/dgemm_blocking_128 src/main.c src/dgemm.c -lm
	gcc -O3 -DBLOCK_SIZE=256 -o out/dgemm_blocking_256 src/main.c src/dgemm.c -lm
	gcc -O3 -DBLOCK_SIZE=512 -o out/dgemm_blocking_512 src/main.c src/dgemm.c -lm

.PHONY: dgemm_avx256_unroll
dgemm_avx256_unroll: 
	gcc -O3 -mavx2 -DUNROLL=2  -o out/dgemm_avx256_unroll_02 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2 -DUNROLL=4  -o out/dgemm_avx256_unroll_04 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2 -DUNROLL=8  -o out/dgemm_avx256_unroll_08 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2 -DUNROLL=16 -o out/dgemm_avx256_unroll_16 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2 -DUNROLL=32 -o out/dgemm_avx256_unroll_32 src/main.c src/dgemm.c -lm

.PHONY: dgemm_avx256_blocking
dgemm_avx256_blocking: 
	gcc -O3 -mavx2 -DBLOCK_SIZE=32  -o out/dgemm_avx256_blocking_032 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2 -DBLOCK_SIZE=64  -o out/dgemm_avx256_blocking_064 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2 -DBLOCK_SIZE=128 -o out/dgemm_avx256_blocking_128 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2 -DBLOCK_SIZE=256 -o out/dgemm_avx256_blocking_256 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx2 -DBLOCK_SIZE=512 -o out/dgemm_avx256_blocking_512 src/main.c src/dgemm.c -lm

.PHONY: dgemm_avx512_unroll
dgemm_avx512_unroll: 
	gcc -O3 -mavx512f -DUNROLL=2  -o out/dgemm_avx512_unroll_02 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -DUNROLL=4  -o out/dgemm_avx512_unroll_04 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -DUNROLL=8  -o out/dgemm_avx512_unroll_08 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -DUNROLL=16 -o out/dgemm_avx512_unroll_16 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -DUNROLL=32 -o out/dgemm_avx512_unroll_32 src/main.c src/dgemm.c -lm

.PHONY: dgemm_avx512_blocking
dgemm_avx512_blocking: 
	gcc -O3 -mavx512f -DBLOCK_SIZE=32  -o out/dgemm_avx512_blocking_032 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -DBLOCK_SIZE=64  -o out/dgemm_avx512_blocking_064 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -DBLOCK_SIZE=128 -o out/dgemm_avx512_blocking_128 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -DBLOCK_SIZE=256 -o out/dgemm_avx512_blocking_256 src/main.c src/dgemm.c -lm
	gcc -O3 -mavx512f -DBLOCK_SIZE=512 -o out/dgemm_avx512_blocking_512 src/main.c src/dgemm.c -lm

.PHONY: clean
clean:
	rm -fr ./out/*
	rm -fr ./out_csv/*
	rm -fr ./out_csv/raw/*

.PHONY: csv
csv: clean all
	exec scripts/csv.sh ./out
