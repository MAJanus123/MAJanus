class Config:
    def __init__(self, workload):
        self.time_interval = 10
        self.cloudhost = "10.12.0.85"
        self.cloudport = 1087   # 云的listen端口
        self.edgehost = "10.12.0.196"
        self.mqhost = "10.12.0.85"
        self.mqport = 5672
        self.mquser = "guest"
        self.mqpw = "guest"
        self.sudopw = "930718930718"
        self.devicetype = 'CPU'
        self.clouddeviceid = '00'
        self.edgedeviceid = '07'
        self.stream_time = 1
        self.org = 'edge_cloud'
        self.influxdb_port = 8086
        self.bucker_name = 'monitor'
        self.INFLUX_TOKEN = 'FFfAmW00J76_vxJm8SrMKhF2qK5nIczFsJh1JAONsa2n1ztTqmj9UlRvjpn4YCUmCrjpt8uSoM4c1MOx2iQT_A=='

        self.workload = workload
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def _get_args(self):
        return self.__dict__
