class virus_obj:
    def __init__(self):
        self.clnp_fn = None
        self.nu_fn = None
        self.q_fn = None
        self.clnp= clnp()
        self.cbar=None
        self.BasisFunctionType=None
        self.R1=None
        self.R2=None
        self.nu=None
        self.q=None
        self.unique_lp=None
        self.map_unique2lp=None
        self.map_Ip2unique=None
        self.Htable=None
class clnp:
    def __init__(self):
        self.l=None
        self.n=None
        self.p=None
        self.optflag=None
        self.c=None
