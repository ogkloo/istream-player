class QoeInfo:
    def __init__(self):
        # Type e.g. 'segment'
        self.type = None

        # Store lastBitrate for calculation of bitrateSwitchWSum
        self.lastBitrate = None

        # Weights for each QoE factor
        self.weights = {
            "bitrateReward": None,
            "bitrateSwitchPenalty": None,
            "rebufferPenalty": None,
            "latencyPenalty": None,
            "playbackSpeedPenalty": None,
        }

        # Weighted Sum for each QoE factor
        self.bitrateWSum = 0  # kbps
        self.bitrateSwitchWSum = 0  # kbps
        self.rebufferWSum = 0  # seconds
        self.latencyWSum = 0  # seconds
        self.playbackSpeedWSum = 0  # e.g. 0.95, 1.0, 1.05

        # Store total QoE value based on current Weighted Sum values
        self.totalQoe = 0
