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
	# ./out/dgemm_avx256 4096

dgemm_avx512:
	gcc -O3 -mavx512f -o out/dgemm_avx512 dgemm_avx512.c
	./out/dgemm_avx512 128
	./out/dgemm_avx512 256
	./out/dgemm_avx512 512
	./out/dgemm_avx512 1024
	./out/dgemm_avx512 2048
	# ./out/dgemm_avx512 4096

clean:
	rm -r ./out/*
