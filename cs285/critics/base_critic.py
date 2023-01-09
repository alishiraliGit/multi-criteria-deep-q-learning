import abc


class BaseCritic(abc.ABC):
    @abc.abstractmethod
    def get_actor_class(self):
        pass

    @abc.abstractmethod
    def update(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        pass
