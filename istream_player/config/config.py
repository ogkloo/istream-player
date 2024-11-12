from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Prediction():
    # Bandwidth before event occurs
    bw_old: int

    # Bandwidth after event occurs
    bw_new: int 

    # This is time we use to set up initial conditions
    #ready_time: float

    # How long until event occurs
    time_to_event: float

    # How long the event is predicted to last
    duration: float

    last_valid_duration: float

    #@classmethod
    #def load_from_json(cls, json: dict):
    #    return cls(json["bw_old"], json["bw_new"], json["time_to_event"], json["duration"])

    @classmethod
    def load_from_string(cls, message):
        # Unless you're me: Don't use this one. The other one is far more sane.
        m = map(float, message.split(','))
        return cls(*list(m))
    
    def download_time(self, file_size, t, log=None):
        if log:
            log.info(f'{file_size=}, {t=}, {self.time_to_event * self.bw_old}')
        if t < self.time_to_event:
            if file_size < (self.time_to_event-t) * self.bw_old:
                if log:
                    log.info(f'finish before event: {file_size=}, {(self.time_to_event-t) * self.bw_old}')
                return file_size/(self.bw_old)
            else:
                #if file_size < (self.time_to_event * self.bw_old) + (self.last_valid_duration * self.bw_new):
                if log:
                    log.info(f'after event: {(file_size-((self.time_to_event-t) * self.bw_old))/(self.bw_new) + self.duration}')

                return (file_size-((self.time_to_event-t) * self.bw_old))/(self.bw_new) + self.duration
        elif t < self.time_to_event + self.duration:
            time_in_outage = (self.time_to_event + self.duration) - t
            return file_size/(self.bw_new) + time_in_outage
        else:
            return file_size/(self.bw_new)
    
    def total_resources(self):
        return self.time_to_event * self.bw_old + self.last_valid_duration * self.bw_new


class StaticConfig(object):
    # Max initial bitrate (bps)
    max_initial_bitrate = 1000000

    # averageSpeed = SMOOTHING_FACTOR * lastSpeed + (1-SMOOTHING_FACTOR) * averageSpeed;
    smoothing_factor = 0.5

    # minimum frame chunk size ratio
    # The size ratio of a segment which is for I-, P-, and B-frames.
    min_frame_chunk_ratio = 0.6

    # VQ threshold
    vq_threshold = 0.8

    # [Not Used] VQ threshold for size ratio
    vq_threshold_size_ratio = min_frame_chunk_ratio * (
        min_frame_chunk_ratio + (1 - min_frame_chunk_ratio) * vq_threshold
    )

    # Update interval
    update_interval = 0.05

    # [Not Used] Chunk size
    chunk_size = 40960

    # [Not Used] Timeout max ratio
    timeout_max_ratio = 2

    # [Not Used] Min Duration for quality increase (ms)
    min_duration_for_quality_increase_ms = 6000

    # [Not Used] Max duration for quality decrease (ms)
    max_duration_for_quality_decrease_ms = 8000

    # [Not Used] Min duration to retrain after discard (ms)
    min_duration_to_retrain_after_discard_ms = 8000

    # [Not Used] Bandwidth fraction
    bandwidth_fraction = 0.75

    # If the packet arrives later than this it should not be consider in bw estimation
    max_packet_delay = 2

    # Continuous bw estimation window (s)
    cont_bw_window = 10


@dataclass
class PlayerConfig:
    # TODO: Move static configurations to dynamic
    static = StaticConfig

    # Required config
    input: str = ""
    run_dir: str = ""

    # Multiplied by update interval in many cases
    time_factor: float = 1

    # Modules
    mod_mpd: str = "mpd"
    mod_downloader: str = "auto"
    mod_bw: str = "bw_meter"
    mod_abr: str = "dash"
    mod_scheduler: str = "scheduler"
    mod_buffer: str = "buffer_manager"
    mod_player: str = "dash"
    mod_analyzer: list[str] = field(default_factory=lambda: ["data_collector"])

    # Buffer Configuration
    buffer_duration: float = 5.0
    safe_buffer_level: float = 1.0
    panic_buffer_level: float = 0.5
    min_rebuffer_duration: float = 1
    min_start_duration: float = 1

    select_as: str = "-"

    ssl_keylog_file: Optional[str] = None

    # Live event logs file path
    live_log: Optional[str] = None

    #predicted_events: List[Prediction] = []

    initial_buffer = 5.0
    initial_quality = 4

    search_method = 'exhaustive'

    def validate(self) -> None:
        """Assert if config properties are set properly"""
        assert bool(
            self.input
        ), "A non-empty '--input' arg or 'input' config is required"
