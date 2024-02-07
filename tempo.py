#plot 01
# Número de soluções encontradas

import matplotlib.pyplot as plt

tempo1= [0.22721576690673828, 0.39221954345703125, 0.6217763423919678, 0.944983959197998, 1.4217870235443115, 1.839165449142456, 2.3603038787841797, 2.8623123168945312, 3.339963912963867, 3.8852896690368652, 4.557690620422363, 5.255352973937988]
tempo2= [6.328571258167431, 5.799918057496437, 6.791044056918455, 6.090489334253089, 7.376978123482922, 6.69951226782604, 8.238642384999146, 7.701619945607258, 7.099605701990032, 7.823274801035397, 8.227415738628445, 9.004672314789602]


num_solucoes = list(range(1, len(tempo1) + 1))

# Plotagem do gráfico
plt.figure()
plt.plot(num_solucoes, tempo1, marker='o', linestyle='-', color='b')
plt.plot(num_solucoes, tempo2, marker='o', linestyle='-', color='r')

plt.xlabel('Número de Soluções Encontradas')
plt.ylabel('Tempo (s)')
plt.title('Soluções Encontradas')
plt.grid(True)
plt.show()