from user import user # importa a classe user

def start_user(total_number_of_users, data):

    # Taxa de dados mínima exigida pelo usuário
    user_required_data_rate = 400 * 1024 #Taxa requerida

    # Lista para armazenar as instâncias dos usuários
    users = []

    # Criação de usuários
    for i in range(0, total_number_of_users):
        new_user = user() # cria um novo usuários
        new_user.id = i # ID do usuário
        new_user.x = data.X1[i] # Posição dos usuários no eixo x
        new_user.y = data.X2[i] # Posição dos usuários no eixo y
        new_user.required_data_rate = user_required_data_rate # taxa de dados mínima exigida
        users.append(new_user) #adiciona o novo usuário na lista
        
    return users