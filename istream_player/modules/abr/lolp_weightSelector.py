from itertools import product
from istream_player.modules.abr.lolp_neuron import Neuron
from istream_player.modules.abr.lolp_qoeEvaluator import LoLpQoeEvaluator


class LolpWeightSelector:
    def __init__(self) -> None:
        self._resetInitialSettings()
        self.targetLatency = 0
        self.bufferMin = 0
        self.segmentDuration = 1
        self.qoeEvaluator = LoLpQoeEvaluator()
        self.instance = {
            "valueList": self.valueList,
            "weightTypeCount": self.weightTypeCount,
            "weightOptions": self.weightOptions,
            "previousLatency": self.previousLatency,
        }

    def _resetInitialSettings(self):
        self.valueList = [0.2, 0.4, 0.6, 0.8, 1]
        self.weightTypeCount = 4
        self.weightOptions = self._getPermutations(self.valueList, self.weightTypeCount)
        self.previousLatency = 0

    @staticmethod
    def _getPermutations(lst, length):
        return list(product(lst, repeat=length))

    def findWeightVector(
        self,
        neurons,
        currentLatency,
        currentBuffer,
        currentRebuffer,
        currentThroughput,
        playbackRate,
    ):
        maxQoE = None
        winnerWeights = None
        winnerBitrate = None
        deltaLatency = abs(currentLatency - self.previousLatency)

        for neuron in neurons:
            for weightVector in self.weightOptions:
                weightsObj = {
                    "throughput": weightVector[0],
                    "latency": weightVector[1],
                    "buffer": weightVector[2],
                    "switch": weightVector[3],
                }

                downloadTime = (
                    neuron["bitrate"] * self.segmentDuration
                ) / currentThroughput
                nextBuffer = self.getNextBuffer(currentBuffer, downloadTime)
                rebuffer = max(0.00001, (downloadTime - nextBuffer))

                wt = 10 if weightsObj["buffer"] == 0 else (1 / weightsObj["buffer"])
                weightedRebuffer = wt * rebuffer

                wt = 10 if weightsObj["latency"] == 0 else (1 / weightsObj["latency"])
                weightedLatency = wt * neuron["state"]["latency"]

                totalQoE = self.qoeEvaluator.calculateSingleUseQoe(
                    neuron["bitrate"], weightedRebuffer, weightedLatency, playbackRate
                )
                if maxQoE is None or totalQoE > maxQoE:
                    maxQoE = totalQoE
                    winnerWeights = weightVector
                    winnerBitrate = neuron["bitrate"]

        if winnerWeights is None and winnerBitrate is None:
            winnerWeights = -1

        self.previousLatency = currentLatency
        return winnerWeights

    def getMinBuffer(self):
        return self.bufferMin

    def getSegmentDuration(self):
        return self.segmentDuration

    def getNextBufferWithBitrate(
        self, bitrateToDownload, currentBuffer, currentThroughput
    ):
        downloadTime = (bitrateToDownload * self.segmentDuration) / currentThroughput
        return self.getNextBuffer(currentBuffer, downloadTime)

    def getNextBuffer(self, currentBuffer, downloadTime):
        nextBuffer = (
            currentBuffer - self.segmentDuration
            if downloadTime > self.segmentDuration
            else currentBuffer + self.segmentDuration - downloadTime
        )
        return nextBuffer
