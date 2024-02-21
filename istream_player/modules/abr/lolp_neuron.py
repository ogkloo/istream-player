class Neuron:
    def __init__(
        self, qualityIndex, bitrate, throughput, latency=0, rebuffer=0, switch=0
    ):
        self.qualityIndex = qualityIndex
        self.bitrate = bitrate
        self.state = {
            "throughput": throughput,
            "latency": latency,
            "rebuffer": rebuffer,
            "switch": switch,
        }

    def update_state(
        self, new_throughput=None, new_latency=None, new_rebuffer=None, new_switch=None
    ):
        if new_throughput is not None:
            self.state["throughput"] = new_throughput
        if new_latency is not None:
            self.state["latency"] = new_latency
        if new_rebuffer is not None:
            self.state["rebuffer"] = new_rebuffer
        if new_switch is not None:
            self.state["switch"] = new_switch
