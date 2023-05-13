python:
	python main.py 16
	python main.py 32
	python main.py 64
	python main.py 128
	python main.py 256
	python main.py 512
	python main.py 1024

dgemm_simple:
	gcc -O3 -o out/dgemm_simple dgemm_simple.c
	./out/dgemm_simple 128
	./out/dgemm_simple 256
	./out/dgemm_simple 512
	./out/dgemm_simple 1024
	./out/dgemm_simple 2048

dgemm_avx256:
	gcc -O3 -mavx2 -o out/dgemm_avx256 dgemm_avx256.c
	./out/dgemm_avx256 128
	./out/dgemm_avx256 256
	./out/dgemm_avx256 512
	./out/dgemm_avx256 1024
	./out/dgemm_avx256 2048

dgemm_avx256_unroll_2:
	gcc -O3 -mavx2 -DUNROLL=2 -o out/dgemm_avx256_unroll_2 dgemm_avx256_unroll.c
	./out/dgemm_avx256_unroll_2 128
	./out/dgemm_avx256_unroll_2 256
	./out/dgemm_avx256_unroll_2 512
	./out/dgemm_avx256_unroll_2 1024
	./out/dgemm_avx256_unroll_2 2048

dgemm_avx256_unroll_4:
	gcc -O3 -mavx2 -DUNROLL=4 -o out/dgemm_avx256_unroll_4 dgemm_avx256_unroll.c
	./out/dgemm_avx256_unroll_4 128
	./out/dgemm_avx256_unroll_4 256
	./out/dgemm_avx256_unroll_4 512
	./out/dgemm_avx256_unroll_4 1024
	./out/dgemm_avx256_unroll_4 2048

dgemm_avx256_unroll_8:
	gcc -O3 -mavx2 -DUNROLL=8 -o out/dgemm_avx256_unroll_8 dgemm_avx256_unroll.c
	./out/dgemm_avx256_unroll_8 128
	./out/dgemm_avx256_unroll_8 256
	./out/dgemm_avx256_unroll_8 512
	./out/dgemm_avx256_unroll_8 1024
	./out/dgemm_avx256_unroll_8 2048

dgemm_avx256_unroll_16:
	gcc -O3 -mavx2 -DUNROLL=16 -o out/dgemm_avx256_unroll_16 dgemm_avx256_unroll.c
	./out/dgemm_avx256_unroll_16 128
	./out/dgemm_avx256_unroll_16 256
	./out/dgemm_avx256_unroll_16 512
	./out/dgemm_avx256_unroll_16 1024
	./out/dgemm_avx256_unroll_16 2048

dgemm_avx256_unroll_32:
	gcc -O3 -mavx2 -DUNROLL=32 -o out/dgemm_avx256_unroll_32 dgemm_avx256_unroll.c
	./out/dgemm_avx256_unroll_32 128
	./out/dgemm_avx256_unroll_32 256
	./out/dgemm_avx256_unroll_32 512
	./out/dgemm_avx256_unroll_32 1024
	./out/dgemm_avx256_unroll_32 2048

dgemm_avx256_unroll: dgemm_avx256_unroll_2 dgemm_avx256_unroll_4 dgemm_avx256_unroll_8 dgemm_avx256_unroll_16 dgemm_avx256_unroll_32

dgemm_avx512:
	gcc -O3 -mavx512f -o out/dgemm_avx512 dgemm_avx512.c
	./out/dgemm_avx512 128
	./out/dgemm_avx512 256
	./out/dgemm_avx512 512
	./out/dgemm_avx512 1024
	./out/dgemm_avx512 2048

clean:
	rm -r ./out/*
