class SetGrad():
    # :TODO this class should be moved in another file!
    _mode: bool = True
    def __init__(self, mode: bool = True) -> None:
        self._prev = mode
        SetGrad._mode = mode

    def __enter__(self) -> None:
        self.disable_grad()

    def __exit__(self) -> None:
        self.enable_grad()

    def _disable_grad(self) -> None:
        self._prev = self.mode
        self.mode = False

    def _enable_grad(self) -> None:
        self._prev = self.mode
        self.mode = True

    @staticmethod
    def disable_grad() -> None:
        SetGrad.mode = False

    @staticmethod
    def enable_grad() -> None:
        SetGrad.mode = True

    @staticmethod
    def is_grad_enabled() -> bool:
        return SetGrad._mode