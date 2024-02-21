from istream_player.modules.abr.lolp_qoeInfo import QoeInfo


class LoLpQoeEvaluator:
    def __init__(self) -> None:
        self._resetInitialSettings()

    def _resetInitialSettings(self):
        self.voPerSegmentQoeInfo = None
        self.segmentDuration = None
        self.maxBitrateKbps = None
        self.minBitrateKbps = None

    def setupPerSegmentQoe(self, sDuration, maxBrKbps, minBrKbps):
        self.voPerSegmentQoeInfo = self._createQoeInfo(
            "segment", sDuration, maxBrKbps, minBrKbps
        )
        self.segmentDuration = sDuration
        self.maxBitrateKbps = maxBrKbps
        self.minBitrateKbps = minBrKbps

    def _createQoeInfo(
        self, fragmentType, fragmentDuration, maxBitrateKbps, minBitrateKbps
    ):
        self.qoeInfo = QoeInfo()
        self.qoeInfo.type = fragmentType

        # Set weight: bitrateReward
        self.qoeInfo.weights["bitrateReward"] = (
            1 if not fragmentDuration else fragmentDuration
        )

        # Set weight: bitrateSwitchPenalty
        self.qoeInfo.weights["bitrateSwitchPenalty"] = 1
        # qoeInfo.weights["bitrateSwitchPenalty"] = 0.02

        # Set weight: rebufferPenalty
        self.qoeInfo.weights["rebufferPenalty"] = (
            1000 if not maxBitrateKbps else maxBitrateKbps
        )

        # Set weight: latencyPenalty
        self.qoeInfo.weights["latencyPenalty"].append(
            {"threshold": 1.1, "penalty": minBitrateKbps * 0.05}
        )

        # Adding the second (catch-all) condition for latency penalty
        self.qoeInfo.weights["latencyPenalty"].append(
            {
                "threshold": 100000000,  # A very high threshold effectively acts as a catch-all
                "penalty": maxBitrateKbps * 0.1,
            }
        )

        # Set weight: playbackSpeedPenalty
        self.qoeInfo.weights["playbackSpeedPenalty"] = (
            200 if not minBitrateKbps else minBitrateKbps
        )
        return self.qoeInfo

    def logSegmentMetrics(
        self, segmentBitrate, segmentRebufferTime, currentLatency, currentPlaybackSpeed
    ):
        if self.voPerSegmentQoeInfo:
            self._logMetricsInQoeInfo(
                segmentBitrate,
                segmentRebufferTime,
                currentLatency,
                currentPlaybackSpeed,
                self.voPerSegmentQoeInfo,
            )

    def _logMetricsInQoeInfo(
        self, bitrate, rebufferTime, latency, playbackSpeed, qoeInfo
    ):
        # Update: bitrate Weighted Sum value
        self.qoeInfo.bitrateWSum += self.qoeInfo.weights["bitrateReward"] * bitrate

        # Update: bitrateSwitch Weighted Sum value
        if self.qoeInfo.lastBitrate is not None:
            self.qoeInfo.bitrateSwitchWSum += self.qoeInfo.weights[
                "bitrateSwitchPenalty"
            ] * abs(bitrate - self.qoeInfo.lastBitrate)
        self.qoeInfo.lastBitrate = bitrate

        # Update: rebuffer Weighted Sum value
        self.qoeInfo.rebufferWSum += (
            self.qoeInfo.weights["rebufferPenalty"] * rebufferTime
        )

        # Update: latency Weighted Sum value
        for latencyRange in self.qoeInfo.weights["latencyPenalty"]:
            if latency <= latencyRange["threshold"]:
                self.qoeInfo.latencyWSum += latencyRange["penalty"] * latency
                break

        # Update: playbackSpeed Weighted Sum value
        self.qoeInfo.playbackSpeedWSum += self.qoeInfo.weights[
            "playbackSpeedPenalty"
        ] * abs(1 - playbackSpeed)

        # Update: Total QoE value
        self.qoeInfo.totalQoe = (
            self.qoeInfo.bitrateWSum
            - self.qoeInfo.bitrateSwitchWSum
            - self.qoeInfo.rebufferWSum
            - self.qoeInfo.latencyWSum
            - self.qoeInfo.playbackSpeedWSum
        )

    def getPerSegmentQoe(self):
        return self.voPerSegmentQoeInfo

    def add_latency_penalty_conditions(self, minBitrateKbps, maxBitrateKbps):
        # Adding the first condition for latency penalty
        self.weights["latencyPenalty"].append(
            {"threshold": 1.1, "penalty": minBitrateKbps * 0.05}
        )

        # Adding the second (catch-all) condition for latency penalty
        self.weights["latencyPenalty"].append(
            {
                "threshold": 100000000,  # A very high threshold effectively acts as a catch-all
                "penalty": maxBitrateKbps * 0.1,
            }
        )

    def calculateSingleUseQoe(
        self, segmentBitrate, segmentRebufferTime, currentLatency, currentPlaybackSpeed
    ):
        singleUseQoeInfo = None

        if self.segmentDuration and self.maxBitrateKbps and self.minBitrateKbps:
            singleUseQoeInfo = self._createQoeInfo(
                "segment",
                self.segmentDuration,
                self.maxBitrateKbps,
                self.minBitrateKbps,
            )

        if singleUseQoeInfo:
            self._logMetricsInQoeInfo(
                segmentBitrate,
                segmentRebufferTime,
                currentLatency,
                currentPlaybackSpeed,
                singleUseQoeInfo,
            )
            return singleUseQoeInfo.totalQoe
        else:
            # Something went wrong
            return 0
