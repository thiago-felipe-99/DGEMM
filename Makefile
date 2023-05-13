python:
	python main.py 16
	python main.py 32
	python main.py 64
	python main.py 128
	python main.py 256
	python main.py 512
	python main.py 1024

dgemm_simple:
	gcc -o out/dgemm_simple dgemm_simple.c
	./out/dgemm_simple 128
	./out/dgemm_simple 256
	./out/dgemm_simple 512
	./out/dgemm_simple 1024
	./out/dgemm_simple 2048
	./out/dgemm_simple 4096
	./out/dgemm_simple 8192

clean:
	rm -r ./out/*
