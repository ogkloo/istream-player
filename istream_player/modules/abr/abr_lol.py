from typing import Dict
import logging
import math
import numpy as np
import random
from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet
from istream_player.models.mpd_objects import Representation
from istream_player.modules.bw_meter.bandwidth import BandwidthMeter
from istream_player.core.buffer import BufferManager
from istream_player.config.config import PlayerConfig

from istream_player.modules.analyzer.analyzer import PlaybackAnalyzer

@ModuleOption("lol", requires=[PlaybackAnalyzer])
class LolpABRController(Module, ABRController):        
    def _manualWeightSelection(self):
        self.weights  = [0.4, 0.4, 0.4, 0.4]
        
    def _randomWeightSelection(self, somElements):
        self.weights = self._getXavierWeights(len(somElements), 4)

    def _getXavierWeights(self, neuronCount, weightCount):
        W = []
        upperBound = (2 / neuronCount) ** 0.5
        for _ in range(weightCount):
            W.append(random.random() * upperBound)

        return W
    
    def _dynamicWeightSelection(self, dynamicWeightsSelector, somElements, currentLatency, currentBuffer, rebuffer, currentThroughput, playbackRate):
        if not self.weights:
            self.weights = self.sortedCenters[-1]
        weightVector = dynamicWeightsSelector.findWeightVector(somElements, currentLatency, currentBuffer, rebuffer, currentThroughput, playbackRate)
        if weightVector is not None and weightVector != -1:
            self.weights = weightVector
    
    def __init__(self, config=None):                       
        self.logger = logging.getLogger(__name__)
        self._reset_initial_settings()

    def _reset_initial_settings(self):
        self.somBitrateNeurons = None
        self.bitrateNormalizationFactor = 1
        self.latencyNormalizationFactor = 100
        self.minBitrate = 0
        self.weights = None
        self.sortedCenters = None
        self.WEIGHT_SELECTION_MODES = {
            'MANUAL': self._manualWeightSelection,
            'RANDOM': self._randomWeightSelection,
            'DYNAMIC': self._dynamicWeightSelection
        }
        self.weightSelectionMode = 'DYNAMIC' # selection of weight updating mode
        self.selectWeightMode()
        
    def selectWeightMode(self):
        """dynamic select the weight selection mode
        """        
        mode_method = self.WEIGHT_SELECTION_MODES.get(self.weightSelectionMode)
        if mode_method:
            mode_method()
            
    async def setup(self, config: PlayerConfig, bandwidth_meter: BandwidthMeter, buffer_manager: BufferManager,):
        # These would need to be replaced or implemented according to the Python environment
        self.buffer_size = config.buffer_duration
        self.bandwidth_meter = bandwidth_meter
        self.buffer_manager = buffer_manager
        self._reset_initial_settings()
    
    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Dict[int, int]:
        final_selections = dict()
        
        # Only use 70% of measured bandwidth
        available_bandwidth = int(self.bandwidth_meter.bandwidth)
        
        def has_seg_id(rep: Representation):
            for seg_id, _ in rep.segments.items():
                if seg_id == index:
                    return True
            return False
        return

    def _getMaxThroughput(self):
        maxThroughput = 0        
        if self.somBitrateNeurons:
            for neuron in self.somBitrateNeurons:
                if neuron['state']['throughput'] > maxThroughput:
                    maxThroughput = neuron['state']['throughput']
        return maxThroughput
    
    @classmethod 
    def _getMagnitude(w):
        magnitude = sum(x**2 for x in w)
        return math.sqrt(magnitude)
    
    @classmethod 
    def _getDistance(a, b, w):
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
        a_state = [a['state']['throughput'], a['state']['latency'], a['state']['rebuffer'], a['state']['switch']]
        b_state = [b['state']['throughput'], b['state']['latency'], b['state']['rebuffer'], b['state']['switch']]
        return __class__._getDistance(a_state, b_state, [1, 1, 1, 1])

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
            neighbourhood = math.exp(-1 * (neuron_distance ** 2) / (2 * (sigma ** 2)))
            self._updateNeuronState(som_neuron, x, neighbourhood)

    def _updateNeuronState(self, neuron, x, neighbourhood):
        state = neuron['state']
        w = [0.01, 0.01, 0.01, 0.01]  # learning rate

        state['throughput'] = state['throughput'] + (x[0] - state['throughput']) * w[0] * neighbourhood
        state['latency'] = state['latency'] + (x[1] - state['latency']) * w[1] * neighbourhood
        state['rebuffer'] = state['rebuffer'] + (x[2] - state['rebuffer']) * w[2] * neighbourhood
        state['switch'] = state['switch'] + (x[3] - state['switch']) * w[3] * neighbourhood

    def _getDownShiftNeuron(self, current_neuron, current_throughput):
        maxSuitableBitrate = 0
        result = current_neuron

        if self.somBitrateNeurons:
            for n in self.somBitrateNeurons:
                if n['bitrate'] < current_neuron['bitrate'] and n['bitrate'] > maxSuitableBitrate and current_throughput > n['bitrate']:
                    maxSuitableBitrate = n['bitrate']
                    result = n
        return result
    
    def getNextQuality(self, mediaInfo, throughput, latency, bufferSize, playbackRate, currentQualityIndex, dynamicWeightsSelector):
        currentLatency = latency
        currentBuffer = bufferSize
        currentThroughput = throughput

        somElements = self._getSomBitrateNeurons(mediaInfo)
        throughputNormalized = throughput / self.bitrateNormalizationFactor
        if throughputNormalized > 1:
            throughputNormalized = self._getMaxThroughput()
        latency = latency / self.latencyNormalizationFactor

        targetLatency = 0
        targetRebufferLevel = 0
        targetSwitch = 0
        throughputDelta = 10000

        self.logger.debug(f"getNextQuality called throughput:{throughputNormalized} latency:{latency} bufferSize:{bufferSize} currentQualityIndex:{currentQualityIndex} playbackRate:{playbackRate}")

        currentNeuron = somElements[currentQualityIndex]
        downloadTime = (currentNeuron['bitrate'] * dynamicWeightsSelector.getSegmentDuration()) / currentThroughput
        rebuffer = max(0, (downloadTime - currentBuffer))

        if currentBuffer - downloadTime < dynamicWeightsSelector.getMinBuffer():
            self.logger.debug(f"Buffer is low for bitrate= {currentNeuron['bitrate']} downloadTime={downloadTime} currentBuffer={currentBuffer} rebuffer={rebuffer}")
            return self._getDownShiftNeuron(currentNeuron, currentThroughput)['qualityIndex']
        
        if self.weightSelectionMode == self.WEIGHT_SELECTION_MODES['MANUAL']:
            self._manualWeightSelection()
        elif self.weightSelectionMode == self.WEIGHT_SELECTION_MODES['RANDOM']:
            self._randomWeightSelection(somElements)
        elif self.weightSelectionMode == self.WEIGHT_SELECTION_MODES['DYNAMIC']:
            self._dynamicWeightSelection(dynamicWeightsSelector, somElements, currentLatency, currentBuffer, rebuffer, currentThroughput, playbackRate)

        minDistance = None
        minIndex = None
        winnerNeuron = None

        for somNeuron in somElements:
            somNeuronState = somNeuron['state']
            somData = [somNeuronState['throughput'], somNeuronState['latency'], somNeuronState['rebuffer'], somNeuronState['switch']]

            distanceWeights = self.weights[:]
            nextBuffer = dynamicWeightsSelector.getNextBufferWithBitrate(somNeuron['bitrate'], currentBuffer, currentThroughput)
            isBufferLow = nextBuffer < dynamicWeightsSelector.getMinBuffer()
            if isBufferLow:
                self.logger.debug(f"Buffer is low for bitrate={somNeuron['bitrate']} downloadTime={downloadTime} currentBuffer={currentBuffer} nextBuffer={nextBuffer}")
            if somNeuron['bitrate'] > throughput - throughputDelta or isBufferLow:
                if somNeuron['bitrate'] != self.minBitrate:
                    # increase throughput weight to select the smaller bitrate
                    distanceWeights[0] = 100

            distance = self._getDistance(somData, [throughputNormalized, targetLatency, targetRebufferLevel, targetSwitch], distanceWeights)
            if minDistance is None or distance < minDistance:
                minDistance = distance
                minIndex = somNeuron['qualityIndex']
                winnerNeuron = somNeuron
        
        ## update current neuron and the neighbourhood with the calculated QoE
        ## will punish current if it is not picked
        bitrateSwitch = abs(currentNeuron['bitrate'] - winnerNeuron['bitrate']) / self.bitrateNormalizationFactor
        self.updateNeurons(currentNeuron, somElements, [throughputNormalized, latency, rebuffer, bitrateSwitch])
        
        ## update bmu and  neighbours with targetQoE=1, targetLatency=0
        self.updateNeurons(winnerNeuron, somElements, [throughputNormalized, targetLatency, targetRebufferLevel, bitrateSwitch])

        return minIndex

    def _getSomBitrateNeurons(self, mediaInfo):
        if not self.somBitrateNeurons:
            self.somBitrateNeurons = []
            # where to get this infor ??
            bitrateList = mediaInfo['bitrateList']
            
            bitrateVector = [] # array
            self.minBitrate = bitrateList[0]['bandwidth']

            # extract all the bandwidth
            for element in bitrateList:
                bitrateVector.append(element['bandwidth'])
                if element['bandwidth'] < self.minBitrate:
                    self.minBitrate = element['bandwidth']
            self.bitrateNormalizationFactor = self._getMagnitude(bitrateVector)

            # push into neuron info
            for i, element in enumerate(bitrateList):
                neuron = {
                    'qualityIndex': i,
                    'bitrate': element['bandwidth'],
                    'state': {
                        'throughput': element['bandwidth'] / self.bitrateNormalizationFactor,
                        'latency': 0,
                        'rebuffer': 0,
                        'switch': 0
                    }
                }
                self.somBitrateNeurons.append(neuron)

            self.sortedCenters = self._getInitialKmeansPlusPlusCenters(self.somBitrateNeurons)

        return self.somBitrateNeurons
    
    def _getRandomData(self, size):
        dataArray = []

        for _ in range(size):
            data = [
                random.random() * self._getMaxThroughput(),  # throughput
                random.random(),  # latency
                random.random(),  # bufferSize
                random.random()  # switch
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
    
    ## Connect with the iStream
    def choose_ideal_selection_buffer_based(self, adaptation_set):
        return
    
    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Dict[int, int]:
        final_selections = dict()

        for adaptation_set in adaptation_sets.values():
            final_selections[adaptation_set.id] = self.choose_ideal_selection_buffer_based(adaptation_set)

        return final_selections