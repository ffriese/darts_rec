from core import Module, ModuleMeta


class CombinedModuleMeta(ModuleMeta):
    # noinspection PyProtectedMember
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        #instance.connect_submodules()
        return instance


class CombinedModule(Module, metaclass=CombinedModuleMeta):
    def __custom_connect__(self):
        self.connect_submodules()

    def __custom_configure__(self):
        self.configure_submodules()

    def configure_submodules(self):
        raise NotImplementedError("Please Implement this method in subclass")

    def connect_submodules(self):
        raise NotImplementedError("Please Implement this method in subclass")