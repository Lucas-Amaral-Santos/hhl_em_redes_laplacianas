# HHL em redes Laplacianas

## Gera Matrizes
O primeiro arquivo a ser execuatdo é **gera_matrizes**. Este gerará as matrizes nas três topologias descritas no artigo.
    1. Rede
    **grid_graph**
    2. Árvore
    **random_tree**
    3. Erdős–Rényi esparso
    **erdos_renyi_sparse**

Em seguida Monta a matriz de Laplace.
**weighted_laplacian**

Gera o vetor corrente b:
**random_current_vector**

Soluciona classicamente utilizando as Chlesky e Descida de Gradiente (usando numpy mesmo).
**solve_classical**

Calcula o numero condicional da matriz dada:
**condition_number**

O **build_instance** usa todas as anteriores para gerar as matrizes com todas as informações necessárias. Usa o método *np.savez* para salvar as matrizes com todas as informações úteis possíveis.

No inicializador apenas corresmos as dimensões das matrizes que queremos gerar. Lembrando que a grade é 2D.

Por último, exportamos as matrizes em formatio pickle.


## Visualize

Apenas carrega as matrizes e plota algumas informações.

## Solvers Clássico

Implementa GD e Cholesky.

Carrega as matrizes e executa.

## Adaptador Laplaciano

Eu precisei adaptar um pouco o Laplaciano para conseguir executar no HHL.
1. As matrizes de Laplace precisam ter dimensões potência de 2. **eh_potencia_de_2** chaca isso.
2. Laplacianos podem ter autovalores nulos, o que o HHL não aceita. Então, foi feito uma remoção de de kernel que faz a transformação (**remover_nucleo**):

$$
L_{sem\_ nucleo} = L + \gamma P_1
$$

**carregar**: Carrega as matrizes dos arguivos.

**preparar**: Adequa as instâncias para o HHL. Além das duas mencionadas, define t, precision (numero de qubits necessários para represenbtaro autovalor). Também estamos arrendondado os autovalores para rodar, já que o circuito do HHL, só está codificado para inteiros. E noramalizamos tudo, já que circuito quânticos são representados por vetores e operadores de norma 1.


## HHL Laplaciano

Apenas executamos o circuito do HHL, montando subrotina a subrotina. Cada subrotina montada porta a porta.

**qpe_exact**: Monta o Quantum Phase Estimation (QPE).
**build_U_pows_from_exact**: Monta uma lista de potências de $U$'s ($U^{2^0}, U^{2^1}, ..., U^{2^{\text{precision}}}$). Lembrando, o U é a codificação da matriz.
**rotacoes_genericas**: Rotação responsável por inverter os autovalores($\frac{1}{\lambda}$)
**hhl_laplaciano**: Monta todas as subrotinas
**run_simulation**: Roda o circuito em im simulador quântico. Não usei ruído.