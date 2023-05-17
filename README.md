## Como fazer build
Para criar o binário do projeto basta esecutar `make dgemm` ele vai criar o executável
em `out/dgemm`

## Como executar
O programa tem 5 algoritmos básicos: simples, transposta, simd manual, avx256 e avx512

### DGEMM Simples
O DGEMM simple é a forma mais básica de fazer uma multiplicação de matrizes:
```
B = matriz[N][N]
A = matriz[N][N]
C = matriz[N][N]

for i range N:
    for j range N:
        for k range N:
            C[i][j] += A[i][k]*B[k][j]
```
Como usar:
```shell 
out/dgemm -d simple -l N
```

### DGEMM Transposta
O DGEMM transposta usa a transposta de A invés de A, com isso o cache da CPU é 
aproveitado melhor:
```
A = matriz[N][N]
B = matriz[N][N]
C = matriz[N][N]

At = transpor(A)

for i range N:
    for j range N:
        for k range N:
            C[i][j] += At[k][i]*B[k][j]
```
Como usar:
```shell 
out/dgemm -d transpose -l N
```

### DGEMM SIMD Manual
O DGEMM SIMD(Single Instruction Multiple Data) Manual faz uma técnina de otimização 
para calcular vários `k` ao mesmo tempo.
Ele já usa a técnica de otimização da matriz transposta:

```
A = matriz[N][N]
B = matriz[N][N]
C = matriz[N][N]

At = transpor(A)

for i range N:
    for j range N:
        for k range N in step 4:
            C[i][j] += At[k + 0][i]*B[k + 0][j]
            C[i][j] += At[k + 1][i]*B[k + 1][j]
            C[i][j] += At[k + 2][i]*B[k + 2][j]
            C[i][j] += At[k + 3][i]*B[k + 3][j]
```
Como usar:
```shell 
out/dgemm -d simd_manual -l N
```

### DEGEMM AVX256/AVX512
O DGEMM AVX256/AVX512 usa as instruções produzidas pela Intel para realizar SIMD 
em processadores X86 no calculo da matriz: 
```
A = matriz[N][N]
B = matriz[N][N]
C = matriz[N][N]

if(avx256)
    QT_INTRUCTIONS = 4
if(avx512)
    QT_INTRUCTIONS = 8

for i range N:
    for j range N in step QT_INTRUCTIONS:
        for k range N:
            C[i][j + 0] += A[i][k]*B[k][j + 0]
            C[i][j + 1] += A[i][k]*B[k][j + 1]
                        .
                        .
                        .
            C[i][j + QT_INTRUCTIONS] += A[i][k]*B[k][j + QT_INTRUCTIONS]
```
Como usar:
```shell 
out/dgemm -d avx256 -l N
out/dgemm -d avx512 -l N
```
## Otimizações Gerais
### Unroll
Essa técninca permitr que o compilador possa fazer unrolling  de um loop que tenha 
um tamanho fixo na hora da compilação, fazendo com a CPU  use menos as instruções de JUMP:
```
B = matriz[N][N]
A = matriz[N][N]
C = matriz[N][N]

R = 8

for i range N:
    for j range N:
        for k range N:
            for r range R:
                C[i][j] += A[i][k + R]*B[k + R][j]
```
Como usar:
```shell 
out/dgemm -d <algoritmo_base>_unroll -l N
```
### Blocking
Essa técninca permitr que os loop i, j, k rodam em blocos de memórias para aproveitar
o cache da CPU:
```
B = matriz[N][N]
A = matriz[N][N]
C = matriz[N][N]

R = 8
BS = 64

for si range N in step BS:
    for sj range N in step BS:
        for sk range N in step BS:
            for i range N:
                for j range N:
                    for k range N:
                        for r range R:
                            C[i][j] += A[i][k + R]*B[k + R][j]
```
DGEMM roda Blocking + Unroll simultaneamente
Como usar:
```shell 
out/dgemm -d <algoritmo_base>_unroll_blocking -l N
```
### Parallel
Essa técninca permite que que um loop possa rodar em várias threads ao mesmo tempo,
assim possibilitadno usar todas os Cores do Processdor:
```
B = matriz[N][N]
A = matriz[N][N]
C = matriz[N][N]

R = 8
BS = 64

THREADS = N / BS

make THREADS for next loop
for si range N in step BS:
    for sj range N in step BS:
        for sk range N in step BS:
            for i range N:
                for j range N:
                    for k range N:
                        for r range R:
                            C[i][j] += A[i][k + R]*B[k + R][j]
```
DGEMM roda Blocking + Unroll + Parallel simultaneamente
Como usar:
```shell 
out/dgemm -d <algoritmo_base>_unroll_blocking_parallel -l N
```
## Argumentos Adicionais
Rodar vários algoritmos:
```shell 
out/dgemm -d alg1,alg2,alg3 -l N
```
Criar matrizes aletórias:
```shell 
out/dgemm -d alg1,alg2,alg3 -l N -r
```
Ver a matriz resultante de cada algoritmo
```shell 
out/dgemm -d alg1,alg2,alg3 -l N -s
```
## Saída do DGEMM
Saída:
```shell
<nome_algoritmo_1>,<tempo_ms>,<GFLOPS/segundo>
<nome_algoritmo_2>,<tempo_ms>,<GFLOPS/segundo>
<nome_algoritmo_3>,<tempo_ms>,<GFLOPS/segundo>
```
