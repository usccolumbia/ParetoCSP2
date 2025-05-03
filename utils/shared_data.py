class SharedData:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.spaceGroupList = []
            # cls._instance.allSgCount = {key: 0 for key in range(1, 231)}
            cls._instance.minEnergyAfterEachGen = 999.0
            cls._instance.optimalStrucSpaceGroup = 0

            cls._instance.crystalElements = []
            cls._instance.elementsCount = []

            cls.wyck_dict = {}
            cls.max_wyck = 0

            # cls._instance.k = 10
            # cls._instance.top_k_structures = []

        return cls._instance

    # setters
    def set_sg(self, x):
        self.spaceGroupList = x

    # def set_asgc(self, x):
    #     self.allSgCount = x

    def set_meaeg(self, x):
        self.minEnergyAfterEachGen = x

    def set_ossg(self, x):
        self.optimalStrucSpaceGroup = x

    def set_ce(self, x):
        self.crystalElements = x

    def set_ec(self, x):
        self.elementsCount = x

    def set_wd(self, x):
        self.wyck_dict = x

    def set_mw(self, x):
        self.max_wyck = x

    # def set_k(self, x):
    #     self.k = x

    # def set_tks(self, x):
    #     self.top_k_structures = x

    # getters
    def get_sg(self):
        return self.spaceGroupList

    # def get_asgc(self):
    #     return self.allSgCount

    def get_meaeg(self):
        return self.minEnergyAfterEachGen

    def get_ossg(self):
        return self.optimalStrucSpaceGroup

    def get_ce(self):
        return self.crystalElements

    def get_ec(self):
        return self.elementsCount

    def get_wd(self):
        return self.wyck_dict

    def get_mw(self):
        return self.max_wyck

    # def get_k(self):
    #     return self.k

    # def get_tks(self):
    #     return self.tks
