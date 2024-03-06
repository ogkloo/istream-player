from istream_player.core.abr import ABRController
from istream_player.core.module import Module, ModuleOption

from istream_player.modules.abr.lolp_abrController import LolpABRController
from istream_player.modules.abr.lolp_qoeEvaluator import LoLpQoeEvaluator


class LoLpRule:
    def __init__(self) -> None:
        self.learningController = LolpABRController()
        self.qoeEvaluator = LoLpQoeEvaluator()
        pass

    def _resetInitialSettings(self):
        self.learningController._resetInitialSettings()
        self.qoeEvaluator._resetInitialSettings()
