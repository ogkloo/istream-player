from typing import Dict
import logging
import math
import numpy as np
import random
from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.buffer import BufferManager

from istream_player.core.module import Module, ModuleOption
from istream_player.models.mpd_objects import AdaptationSet

from istream_player.modules.analyzer.analyzer import PlaybackAnalyzer

from istream_player.modules.abr.lolp_neuron import Neuron
from istream_player.modules.abr.lolp_weightSelector import LolpWeightSelector


@ModuleOption("lol", requires=[BandwidthMeter, BufferManager, PlaybackAnalyzer])
class LolpABRController(Module, ABRController):
    logger = logging.getLogger("ABRLoLP")

    def __init__(self, config=None):
        self._resetInitialSettings()

    async def setup(
        self,
        config: PlayerConfig,
        bandwidth_meter: BandwidthMeter,
        buffer_manager: BufferManager,
        analyzer: PlaybackAnalyzer,
    ):
        # These would need to be replaced or implemented according to the Python environment
        self.config = config
        self.buffer_manager = buffer_manager
        self.bandwidth_meter = bandwidth_meter
        self.analyzer = analyzer
        self._resetInitialSettings()

        self.dynamicWeightsSelector = LolpWeightSelector()

    def _resetInitialSettings(self):
        self.somBitrateNeurons = None
        self.bitrateNormalizationFactor = 1
        self.latencyNormalizationFactor = 100
        self.minBitrate = 0
        self.weights = None
        self.sortedCenters = None
        self.WEIGHT_SELECTION_MODES = {
            "MANUAL": self._manualWeightSelection,
            "RANDOM": self._randomWeightSelection,
            "DYNAMIC": self._dynamicWeightSelection,
        }
        self.weightSelectionMode = "MANUAL"  # selection of weight updating mode
        self.selectWeightMode()

    def selectWeightMode(self):
        """dynamic select the weight selection mode"""
        mode_method = self.WEIGHT_SELECTION_MODES.get(self.weightSelectionMode)
        if mode_method:
            mode_method()

    def update_selection(
        self, adaptation_sets: Dict[int, AdaptationSet], index: int
    ) -> Dict[int, int]:
        final_selections = dict()
        # Only use 70% of measured bandwidth
        available_bandwidth = int(self.bandwidth_meter.bandwidth)
        buffer_level = self.buffer_manager.buffer_level

        for adaptation_set in adaptation_sets.values():
            final_selections[adaptation_set.id] = self.getNextQuality(
                adaptation_set,  # mediaInfo
                available_bandwidth,  # throughput
                0,  # latency
                buffer_level,  # bufferSize
                1,  # playbackRate
                0,  # currentQualityIndex
                self.dynamicWeightsSelector,  # dynamicWeightsSelector
            )
        return final_selections

    def _getMaxThroughput(self):
        maxThroughput = 0
        if self.somBitrateNeurons:
            for neuron in self.somBitrateNeurons:
                if neuron.state["throughput"] > maxThroughput:
                    maxThroughput = neuron.state["throughput"]
        return maxThroughput

    def _getMagnitude(self, w):
        magnitude = sum(x**2 for x in w)
        return math.sqrt(magnitude)

    def _getDistance(self, a, b, w):
        sum_squares = sum(w[i] * (a[i] - b[i]) ** 2 for i in range(len(a)))
        sign = -1 if sum_squares < 0 else 1
        return sign * math.sqrt(abs(sum_squares))

    def _getNeuronDistance(self, a, b):
        """_summary_

        Args:
            a (object): array
            b (object): array

        Returns:
            (number): distance between 2 arrays with weights
        """
        # throughput, latency, rebuffer, switch
        a_state = [
            a.state["throughput"],
            a.state["latency"],
            a.state["rebuffer"],
            a.state["switch"],
        ]
        b_state = [
            b.state["throughput"],
            b.state["latency"],
            b.state["rebuffer"],
            b.state["switch"],
        ]
        return self._getDistance(a_state, b_state, [1, 1, 1, 1])

    def updateNeurons(self, winner_neuron, som_elements, x):
        """_summary_

        Args:
            winner_neuron (object): winnerNeuron
            som_elements (array): somElements
            x (_type_): x
        """
        for som_neuron in som_elements:
            sigma = 0.1
            neuron_distance = self._getNeuronDistance(som_neuron, winner_neuron)
            neighbourhood = math.exp(-1 * (neuron_distance**2) / (2 * (sigma**2)))
            self._updateNeuronState(som_neuron, x, neighbourhood)

    def _updateNeuronState(self, neuron: Neuron, x, neighbourhood):
        _state = neuron.state
        w = [0.01, 0.01, 0.01, 0.01]  # learning rate

        _new_throughput = (
            _state["throughput"] + (x[0] - _state["throughput"]) * w[0] * neighbourhood
        )
        _new_latency = (
            _state["latency"] + (x[1] - _state["latency"]) * w[1] * neighbourhood
        )
        _new_rebuffer = (
            _state["rebuffer"] + (x[2] - _state["rebuffer"]) * w[2] * neighbourhood
        )
        _new_switch = (
            _state["switch"] + (x[3] - _state["switch"]) * w[3] * neighbourhood
        )
        neuron.update_state(_new_throughput, _new_latency, _new_rebuffer, _new_switch)

    def _getDownShiftNeuron(self, current_neuron: Neuron, current_throughput):
        maxSuitableBitrate = 0
        result = current_neuron

        if self.somBitrateNeurons:
            for n in self.somBitrateNeurons:
                if (
                    n.bitrate < current_neuron.bitrate
                    and n.bitrate > maxSuitableBitrate
                    and current_throughput > n.bitrate
                ):
                    maxSuitableBitrate = n.bitrate
                    result = n
        return result

    def getNextQuality(
        self,
        mediaInfo,
        throughput,
        latency,
        bufferSize,
        playbackRate,
        currentQualityIndex,
        dynamicWeightsSelector,
    ):
        currentLatency = latency
        currentBuffer = self.buffer_manager.buffer_level
        currentThroughput = self.bandwidth_meter.bandwidth

        somElements = self._getSomBitrateNeurons(mediaInfo)
        throughputNormalized = throughput / self.bitrateNormalizationFactor
        if throughputNormalized > 1:
            throughputNormalized = self._getMaxThroughput()
        latency = latency / self.latencyNormalizationFactor

        targetLatency = 0
        targetRebufferLevel = 0
        targetSwitch = 0
        throughputDelta = 10000

        self.logger.debug(
            f"getNextQuality called throughput:{throughputNormalized} latency:{latency} bufferSize:{bufferSize} currentQualityIndex:{currentQualityIndex} playbackRate:{playbackRate}"
        )

        currentNeuron = somElements[currentQualityIndex]
        downloadTime = (
            currentNeuron.bitrate * dynamicWeightsSelector.getSegmentDuration()
        ) / currentThroughput
        rebuffer = max(0, (downloadTime - currentBuffer))

        if currentBuffer - downloadTime < dynamicWeightsSelector.getMinBuffer():
            self.logger.debug(
                f"Buffer is low for bitrate= {currentNeuron.bitrate} downloadTime={downloadTime} currentBuffer={currentBuffer} rebuffer={rebuffer}"
            )
            return self._getDownShiftNeuron(
                currentNeuron, currentThroughput
            ).qualityIndex

        if self.weightSelectionMode == self.WEIGHT_SELECTION_MODES["MANUAL"]:
            self._manualWeightSelection()
        elif self.weightSelectionMode == self.WEIGHT_SELECTION_MODES["RANDOM"]:
            self._randomWeightSelection(somElements)
        elif self.weightSelectionMode == self.WEIGHT_SELECTION_MODES["DYNAMIC"]:
            self._dynamicWeightSelection(
                dynamicWeightsSelector,
                somElements,
                currentLatency,
                currentBuffer,
                rebuffer,
                currentThroughput,
                playbackRate,
            )

        minDistance = None
        minIndex = None
        winnerNeuron = None

        for somNeuron in somElements:
            somNeuronState = somNeuron.state
            somData = [
                somNeuronState["throughput"],
                somNeuronState["latency"],
                somNeuronState["rebuffer"],
                somNeuronState["switch"],
            ]

            distanceWeights = self.weights
            nextBuffer = dynamicWeightsSelector.getNextBufferWithBitrate(
                somNeuron.bitrate, currentBuffer, currentThroughput
            )
            isBufferLow = nextBuffer < dynamicWeightsSelector.getMinBuffer()
            if isBufferLow:
                self.logger.debug(
                    f"Buffer is low for bitrate={somNeuron.bitrate} downloadTime={downloadTime} currentBuffer={currentBuffer} nextBuffer={nextBuffer}"
                )
            if somNeuron.bitrate > throughput - throughputDelta or isBufferLow:
                if somNeuron.bitrate != self.minBitrate:
                    # increase throughput weight to select the smaller bitrate
                    distanceWeights[0] = 100

            distance = self._getDistance(
                somData,
                [
                    throughputNormalized,
                    targetLatency,
                    targetRebufferLevel,
                    targetSwitch,
                ],
                distanceWeights,
            )
            if minDistance is None or distance < minDistance:
                minDistance = distance
                minIndex = somNeuron.qualityIndex
                winnerNeuron = somNeuron

        ## update current neuron and the neighbourhood with the calculated QoE
        ## will punish current if it is not picked
        bitrateSwitch = (
            abs(currentNeuron.bitrate - winnerNeuron.bitrate)
            / self.bitrateNormalizationFactor
        )
        self.updateNeurons(
            currentNeuron,
            somElements,
            [throughputNormalized, latency, rebuffer, bitrateSwitch],
        )

        ## update bmu and  neighbours with targetQoE=1, targetLatency=0
        self.updateNeurons(
            winnerNeuron,
            somElements,
            [throughputNormalized, targetLatency, targetRebufferLevel, bitrateSwitch],
        )

        return minIndex

    def _manualWeightSelection(self):
        self.weights = [0.4, 0.4, 0.4, 0.4]

    def _randomWeightSelection(self, somElements):
        self.weights = self._getXavierWeights(len(somElements), 4)

    def _getXavierWeights(self, neuronCount, weightCount):
        W = []
        upperBound = (2 / neuronCount) ** 0.5
        for _ in range(weightCount):
            W.append(random.random() * upperBound)
        return W

    def _dynamicWeightSelection(
        self,
        dynamicWeightsSelector,
        somElements,
        currentLatency,
        currentBuffer,
        rebuffer,
        currentThroughput,
        playbackRate,
    ):
        if not self.weights:
            self.weights = self.sortedCenters[-1]
        weightVector = dynamicWeightsSelector.findWeightVector(
            somElements,
            currentLatency,
            currentBuffer,
            rebuffer,
            currentThroughput,
            playbackRate,
        )
        if weightVector is not None and weightVector != -1:
            self.weights = weightVector

    def _getSomBitrateNeurons(self, mediaInfo):
        if not self.somBitrateNeurons:
            self.somBitrateNeurons = []
            # where to get this infor ??
            bitrateList = mediaInfo.representations

            bitrateVector = []  # array
            self.minBitrate = bitrateList[0].bandwidth

            # extract all the bandwidth
            for i in bitrateList:
                bitrateVector.append(bitrateList[i].bandwidth)
                if bitrateList[i].bandwidth < self.minBitrate:
                    self.minBitrate = bitrateList[i].bandwidth
            self.bitrateNormalizationFactor = self._getMagnitude(bitrateVector)

            # push into neuron info
            for i, _j in enumerate(bitrateList):
                _throughput = (
                    bitrateList[_j].bandwidth / self.bitrateNormalizationFactor
                )
                _latency = 0
                _rebuffer = 0
                _switch = 0

                neuron = Neuron(
                    i,
                    bitrateList[_j].bandwidth,
                    _throughput,
                    _latency,
                    _rebuffer,
                    _switch,
                )
                self.somBitrateNeurons.append(neuron)

            self.sortedCenters = self._getInitialKmeansPlusPlusCenters(
                self.somBitrateNeurons
            )

        return self.somBitrateNeurons

    def _getRandomData(self, size):
        dataArray = []

        for _ in range(size):
            data = [
                random.random() * self._getMaxThroughput(),  # throughput
                random.random(),  # latency
                random.random(),  # bufferSize
                random.random(),  # switch
            ]
            dataArray.append(data)
        return dataArray

    def _getInitialKmeansPlusPlusCenters(self, somElements):
        centers = []
        randomDataSet = self._getRandomData(len(somElements) ** 2)
        centers.append(randomDataSet[0])
        distanceWeights = [1, 1, 1, 1]

        for k in range(1, len(somElements)):
            nextPoint = None
            maxDistance = None
            for i in range(len(randomDataSet)):
                currentPoint = randomDataSet[i]
                minDistance = None
                for center in centers:
                    distance = self._getDistance(currentPoint, center, distanceWeights)
                    if minDistance is None or distance < minDistance:
                        minDistance = distance
                if maxDistance is None or minDistance > maxDistance:
                    nextPoint = currentPoint
                    maxDistance = minDistance
            centers.append(nextPoint)

        # find the least similar center
        maxDistance = None
        leastSimilarIndex = None
        for i in range(len(centers)):
            distance = 0
            for j in range(len(centers)):
                if i == j:
                    continue
                distance += self._getDistance(centers[i], centers[j], distanceWeights)
            if maxDistance is None or distance > maxDistance:
                maxDistance = distance
                leastSimilarIndex = i

        # Move centers to sortedCenters
        sortedCenters = []
        sortedCenters.append(centers.pop(leastSimilarIndex))
        while centers:
            minDistance = None
            minIndex = None
            for i, center in enumerate(centers):
                distance = self._getDistance(sortedCenters[0], center, distanceWeights)
                if minDistance is None or distance < minDistance:
                    minDistance = distance
                    minIndex = i
            sortedCenters.append(centers.pop(minIndex))

        return sortedCenters
