.PHONY: all
all: dgemm dgemm_unroll dgemm_blocking

.PHONY: prepare
prepare:
	mkdir -p ./out
	mkdir -p ./out_csv/raw

.PHONY: dgemm
dgemm: prepare
	gcc  -O3 -fopenmp -fopenmp -march=native -o out/dgemm src/main.c src/dgemm.c -lm

.PHONY: dgemm_unroll
dgemm_unroll: 
	gcc  -O3 -fopenmp -march=native -DUNROLL=2  -o out/dgemm_unroll_02 src/main.c src/dgemm.c -lm
	gcc  -O3 -fopenmp -march=native -DUNROLL=4  -o out/dgemm_unroll_04 src/main.c src/dgemm.c -lm
	gcc  -O3 -fopenmp -march=native -DUNROLL=8  -o out/dgemm_unroll_08 src/main.c src/dgemm.c -lm
	gcc  -O3 -fopenmp -march=native -DUNROLL=16 -o out/dgemm_unroll_16 src/main.c src/dgemm.c -lm

.PHONY: dgemm_blocking
dgemm_blocking: 
	gcc  -O3 -fopenmp -march=native -DBLOCK_SIZE=32  -o out/dgemm_blocking_032 src/main.c src/dgemm.c -lm
	gcc  -O3 -fopenmp -march=native -DBLOCK_SIZE=64  -o out/dgemm_blocking_064 src/main.c src/dgemm.c -lm
	gcc  -O3 -fopenmp -march=native -DBLOCK_SIZE=128 -o out/dgemm_blocking_128 src/main.c src/dgemm.c -lm
	gcc  -O3 -fopenmp -march=native -DBLOCK_SIZE=256 -o out/dgemm_blocking_256 src/main.c src/dgemm.c -lm
	gcc  -O3 -fopenmp -march=native -DBLOCK_SIZE=512 -o out/dgemm_blocking_512 src/main.c src/dgemm.c -lm

.PHONY: clean
clean:
	rm -fr ./out/*
	rm -fr ./out_csv/*
	rm -fr ./out_csv/raw/*

.PHONY: csv_basic
csv_basic: dgemm
	exec scripts/create_csv.sh ./out/dgemm simple,transpose,simd_manual,avx256 '32:1024:32' basic 

.PHONY: csv_unroll
csv_unroll: dgemm_unroll
	exec scripts/create_csv.sh './out/dgemm_unroll_*' simple_unroll,transpose_unroll,simd_manual_unroll,avx256_unroll '32:1024:32' unroll  
                                                                                           
.PHONY: csv_blocking                                                                       
csv_blocking: dgemm_blocking                                                                    
	exec scripts/create_csv.sh './out/dgemm_blocking_*' simple_unroll_blocking,transpose_unroll_blocking,simd_manual_unroll_blocking,avx256_unroll_blocking '32:1024:32' blocking  
                                                                                                                               
.PHONY: csv_parallel                                                                                                           
csv_parallel: dgemm_blocking                                                                                                        
	exec scripts/create_csv.sh './out/dgemm_blocking_*' simple_unroll_blocking_parallel,transpose_unroll_blocking_parallel,simd_manual_unroll_blocking_parallel,avx256_unroll_blocking_parallel '32:1024:32' parallel  
