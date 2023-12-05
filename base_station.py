class base_station:
    def __init__(self):
        self.id = 0
        self.x = 0
        self.y = 0
        self.transmit_power = 0
        self.frequency = 0
        self.bandwidth = 0
        self.total_users = 0
        self.connected_users = 0
        self.base_station_connected = False
        self.total_PRB = 0
        self.remaining_PRB = 0
        self.coverage_area = 0
        self.height = 0