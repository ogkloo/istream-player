'''
    Implements Learn2Adapt roughly as implemented in dash.js.
        const horizon=8//Optimization horizon
        const VL = Math.pow(horizon,0.2);//Cautiousness parameter
        const alpha =Math.max(Math.pow(horizon,0.7),VL*Math.sqrt(horizon));//Step size
'''

# from collections import OrderedDict
from typing import Dict

from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
# from istream_player.core.buffer import BufferManager
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet
from math import pow, sqrt


@ModuleOption("l2a", requires=[BandwidthMeter])
class L2AABRController(Module, ABRController):
    def __init__(self):
        super().__init__()
        self.w = []
        self.prev_w = []
        self.Q = 0
        self.bandwidth_meter = None
        self.buffer_target = 1.5

        # Optimization horizon
        self.horizon = 8
        # Cautiousness
        self.VL = pow(self.horizon, 0.2)
        # Step size
        self.alpha = max(pow(self.horizon, 0.7), self.VL*sqrt(self.horizon))
        # Segment length WARNING: This really should be done better
        self.V = 1

    async def setup(self, config: PlayerConfig, bandwidth_meter: BandwidthMeter):
        self.bandwidth_meter = bandwidth_meter

    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Dict[int, int]:
        available_bandwidth = int(self.bandwidth_meter.bandwidth)

        # Count the number of video adaptation sets and audio adaptation sets
        # "bitrates" in dash.js/L2A implementation
        num_videos = 0
        num_audios = 0
        # Usually there's only one of these
        for adaptation_set in adaptation_sets.values():
            if adaptation_set.content_type == "video":
                num_videos += 1
            else:
                num_audios += 1

        # Calculate ideal selections
        if num_videos == 0 or num_audios == 0:
            # If there are no adaptation sets
            bw_per_adaptation_set = available_bandwidth / (num_videos + num_audios)
            ideal_selection: Dict[int, int] = dict()

            for adaptation_set in adaptation_sets.values():
                ideal_selection[adaptation_set.id] = self.choose_ideal_selection_l2a(
                    adaptation_set, bw_per_adaptation_set
                )
        else:
            # Allocate 80% of bandwidth to video, 20% to audio
            # Since we're testing without audio rn I've changed this to be only video
            bw_per_video = (available_bandwidth * 1) / num_videos
            bw_per_audio = (available_bandwidth * 0) / num_audios
            ideal_selection: Dict[int, int] = dict()
            for adaptation_set in adaptation_sets.values():
                if adaptation_set.content_type == "video":
                    ideal_selection[adaptation_set.id] = self.choose_ideal_selection_l2a(adaptation_set, bw_per_video)
                else:
                    ideal_selection[adaptation_set.id] = self.choose_ideal_selection_l2a(adaptation_set, bw_per_audio)

        return ideal_selection

    def choose_ideal_selection_l2a(self, adaptation_set: AdaptationSet, bw) -> int:
        """
        Choose the ideal bitrate selection for one adaptation_set without caring about the buffer level or
        any other things
        Parameters
        ----------
        adaptation_set
            The adaptation_set to choose
        bw
            The bandwidth could be allocated to this adaptation set
        Returns
        -------
        id: int
            The representation id
        """
        representations = sorted(adaptation_set.representations.values(), key=lambda x: x.bandwidth, reverse=True)
        diff1 = [0]*len(representations)

        # First call
        if len(self.w) == 0:
            # initialize w to zeroes
            self.w = [0]*len(representations)
            self.Q = 0
            # fix weights
            self.w[0] = 0.33
            self.prev_w[0] = 1
            for i in range(1, len(representations)):
                self.w[i] = 0.33
                self.prev_w[i] = 0

        best_bitrate = max(representations, key=lambda r: r.bandwidth)
        # Check units on this one, shd be in mbps
        c_throughput = bw/1000
        # for bitrates
        for representation in representations:
            bitrate = representation.bandwidth/(1000*1000) # conver to mbps
            self.w[i] = self.prev_w - (1/(2*self.alpha)) * (self.V*bitrate) * (self.Q-self.VL)/min(2*best_bitrate, c_throughput)
            diff1[i] = self.w[i] - self.prev_w[i]
        # If there's no representation whose bitrate is lower than the estimate, return the lowest one
        return representations[-1].id
