import random

# Vetor original
vetor_original = [1, 2, 3, 4, 5]

# Contador de repetições
repeticoes = 0

# Loop while para realizar cinco repetições
while repeticoes < 5:
    # Vetor que será preenchido com valores aleatórios
    vetor_preenchido = []

    # Incrementa o contador de repetições
    repeticoes += 1

    # Preencher o vetor_preenchido com valores aleatórios
    for _ in vetor_original:
        valor_aleatorio = random.randint(0, 100)  # Gera um valor aleatório entre 0 e 100
        vetor_preenchido.append(valor_aleatorio)

    # Imprimir o vetor preenchido para esta iteração
    print(f"Iteração {repeticoes}: {vetor_preenchido}")
