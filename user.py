class user:
    def __init__(self):
        self.id = 0
        self.x = 0
        self.y = 0
        self.data_rate = 0
        self.required_data_rate = 0
        self.base_station_id = 0
        self.used_PRBs = 0
        self.CQI = 0
        self.SIRN = 0
        self.user_connected = False # Usuário conectado?
        self.time = 0 # Hora a qual o usuário pertence
        self.base_station_type = 0 # Small = 1 | Macro = 2