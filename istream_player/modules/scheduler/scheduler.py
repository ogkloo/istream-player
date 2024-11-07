import asyncio
import itertools
import logging
import json
import zmq
import math

from time import sleep

from asyncio import Task
from typing import Dict, Optional, Set, Callable
from dataclasses import dataclass

from istream_player.config.config import (PlayerConfig, Prediction)
from istream_player.core.abr import ABRController
from istream_player.core.buffer import BufferManager
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.downloader import (DownloadManager, DownloadRequest,
                                            DownloadType)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import AdaptationSet
from istream_player.utils import critical_task

RECV_PORT = 5555
ACK_PORT = 5554

def all_products(A, K):
    """
    Generate all products of set A with lengths ranging from 1 to K.
    
    :param A: The set of elements to form products from.
    :param K: The maximum length of the product.
    :return: An iterator that yields tuples representing products of lengths 1 through K.
    """
    return itertools.chain.from_iterable(
        itertools.product(A, repeat=r) for r in range(1, K+1)
    )


@dataclass
class Settings():
    w1: float
    w2: float
    w3: float
    w4: float
    p: Callable 
    q: Callable

@ModuleOption(
    "scheduler", default=True, requires=["segment_downloader", BandwidthMeter, BufferManager, MPDProvider, ABRController]
)
class SchedulerImpl(Module, Scheduler):
    log = logging.getLogger("SchedulerImpl")
    # TODO: Load this from config or whatever
    settings = Settings(1, 5, 20, 20, lambda x: x**2, math.log)

    # Before the notification comes, we need to focus on 
    # exactly meeting our initial conditions.
    notification_received = False

    def __init__(self):
        super().__init__()

        self.adaptation_sets: Optional[Dict[int, AdaptationSet]] = None
        self.started = False

        self._task: Optional[Task] = None
        self._index = 0
        self._representation_initialized: Set[str] = set()
        self._current_selections: Optional[Dict[int, int]] = None

        self._end = False
        self._dropped_index = None

    async def setup(
        self,
        config: PlayerConfig,
        segment_downloader: DownloadManager,
        bandwidth_meter: BandwidthMeter,
        buffer_manager: BufferManager,
        mpd_provider: MPDProvider,
        abr_controller: ABRController,
    ):
        self.max_buffer_duration = config.buffer_duration
        self.update_interval = config.static.update_interval
        self.time_factor = config.time_factor

        self.download_manager = segment_downloader
        self.bandwidth_meter = bandwidth_meter
        self.buffer_manager = buffer_manager
        self.abr_controller = abr_controller
        self.mpd_provider = mpd_provider

        # These should default to None or something
        # But right now since I'm using them they're baked in
        self.initial_buffer = config.initial_buffer

        try:
            self.context = zmq.Context()
            self.receiver = self.context.socket(zmq.SUB)
            self.receiver.connect(f"tcp://localhost:{RECV_PORT}")
            self.receiver.setsockopt_string(zmq.SUBSCRIBE, "")
        except:
            raise Exception('zmq error')

        # self.sender = self.context.socket(zmq.PUB)
        # self.sender.bind(f"tcp://localhost:{ACK_PORT}")

        select_as = config.select_as.split("-")
        if len(select_as) == 1 and select_as[0].isdecimal():
            self.selected_as_start = int(select_as[0])
            self.selected_as_end = int(select_as[0])
        elif (
            len(select_as) == 2
            and (select_as[0].isdecimal() or select_as[0] == "")
            and (select_as[1].isdecimal() or select_as[1] == "")
        ):
            self.selected_as_start = int(select_as[0]) if select_as[0] != "" else None
            self.selected_as_end = int(select_as[1]) if select_as[1] != "" else None
        else:
            raise Exception("select_as should be of the format '<uint>-<uint>' or '<uint>'.")

    def segment_limits(self, adap_sets: Dict[int, AdaptationSet]) -> tuple[int, int]:
        ids = [
            [[seg_id for seg_id in repr.segments.keys()] for repr in as_val.representations.values()]
            for as_val in adap_sets.values()
        ]
        ids = itertools.chain(*ids)
        ids = list(itertools.chain(*ids))
        # print(adap_sets, ids)
        return min(ids), max(ids)

    def quantize(self, download_plan):
        quantized_bws = []
        for download in download_plan:
            quantized_download = {}
            for (adaptation_set_id, ideal_bandwidth) in download.items():
                adaptation_set = self.adaptation_sets[adaptation_set_id]
                representations = adaptation_set.representations
                representations = sorted(representations.items(), key=lambda r: r[1].bandwidth)

                # Find ~acceptable bitrate
                self.log.info(f'{representations=}')
                for r in representations:
                    self.log.info(f'{r=}')
                for representation_id, representation in representations:
                    self.log.info(f'{representation_id=}: {representation.bandwidth=}, {ideal_bandwidth=}')
                    if representation.bandwidth > ideal_bandwidth:
                        quantized_download[adaptation_set_id] = representation_id
                        break

                quantized_bws.append(quantized_download)
        return quantized_bws

    @critical_task()
    async def run(self):
        await self.mpd_provider.available()
        assert self.mpd_provider.mpd is not None
        self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)
        # print(f"{self.adaptation_sets=}")

        # Start from the min segment index
        self._index = self.segment_limits(self.adaptation_sets)[0]

        notification_received = False

        # Called as often as possible
        while True:
            # Check buffer level
            notification = await self.check_messages()
            if notification is not None:
                self.log.info(f'Received {notification=} @ {self.buffer_manager.buffer_level}, {self._index=}')
                notification_received = True



            # Check predictions.
            # Notifications are sent via ZeroMQ. To send a notification, use the associated send_notification.py
            # script, or send a message to localhost:5555 containing the JSON of the event.
            if notification is not None:
                notification = ' '.join(notification.split()[1:])
                self.log.info(f'processing {notification=}')
                prediction = Prediction.load_from_string(notification)

                # Warning: Bad code ahead. *2 shouldn't really be here, it's there bc we have 1/2s segment sizes.
                max_K = math.ceil((prediction.time_to_event + prediction.duration + prediction.last_valid_duration) * 2)
                self.log.info(f'{max_K=}')
                adaptation_set = self.adaptation_sets[0]
                representations = adaptation_set.representations
                representations = sorted(representations.items(), key=lambda r: r[1].bandwidth)
                scored_plans = []
                self.log.info('started scoring')
                for plan in all_products(representations, max_K):
                    plan_score = await self.score(prediction, plan)
                    scored_plans.append((plan_score, plan))
                
                best_plan = max(scored_plans, key=lambda x: x[0])
                download_plan = [{0: idx} for idx,repr in best_plan[-1]]
                self.log.info(f'finished scoring, {download_plan=}, {best_plan=}')

                # Download each segment one after another and don't screw up stateful variables
                for selections in download_plan:
                    self.log.info(f'{selections=}')
                    # All adaptation sets take the current bandwidth
                    adap_bw = {as_id: self.bandwidth_meter.bandwidth for as_id in selections.keys()}

                    # Get segments to download for each adaptation set
                    try:
                        segments = {
                            adaptation_set_id: self.adaptation_sets[adaptation_set_id].representations[selection].segments[self._index]
                            for adaptation_set_id, selection in selections.items()
                        }
                    except KeyError:
                        # No more segments left
                        self.log.info("No more segments left")
                        self._end = True
                        return

                    # Download one segment from each adaptation set

                    self.log.info(f"Download starting: {selections=}, {segments=}, {adap_bw=}")

                    for listener in self.listeners:
                        await listener.on_segment_download_start(self._index, adap_bw, segments)
                    
                    # Called ~once per segment (+ reinitializations)
                    urls = []
                    for adaptation_set_id, selection in selections.items():
                        adaptation_set = self.adaptation_sets[adaptation_set_id]
                        representation = adaptation_set.representations[selection]
                        representation_str = "%d:%d" % (adaptation_set_id, representation.id)

                        # Download initial segment
                        if representation_str not in self._representation_initialized:
                            await self.download_manager.download(DownloadRequest(representation.initialization, DownloadType.STREAM_INIT))
                            await self.download_manager.wait_complete(representation.initialization)
                            self.log.info(f"Segment {self._index} Complete. Move to next segment")
                            self._representation_initialized.add(representation_str)

                        # Download next segment
                        try:
                            segment = representation.segments[self._index]
                        except IndexError:
                            self.log.info("Segments ended")
                            self._end = True
                            return
                        urls.append(segment.url)

                        await self.download_manager.download(DownloadRequest(segment.url, DownloadType.SEGMENT))

                    self.log.info(f"Waiting for completion urls {urls}")
                    results = [await self.download_manager.wait_complete(url) for url in urls]
                    self.log.info(f"Completed downloading from urls {urls}")
                    if any([result is None for result in results]):
                        # Result is None means the stream got dropped
                        self._dropped_index = self._index
                        continue
                    download_stats = {as_id: self.bandwidth_meter.get_stats(segment.url) for as_id, segment in segments.items()}
                    for listener in self.listeners:
                        await listener.on_segment_download_complete(self._index, segments, download_stats)
                    self._index += 1
                    await self.buffer_manager.enqueue_buffer(segments)

            else:
                self.log.info("No notification")

                self.log.info(f'{self.buffer_manager.buffer_level=}')
                if notification_received == True:
                    if self.buffer_manager.buffer_level > self.max_buffer_duration:
                        await asyncio.sleep(self.time_factor * self.update_interval)
                        continue
                else:
                    if self.buffer_manager.buffer_level == self.initial_buffer:
                        await asyncio.sleep(self.time_factor * self.update_interval)
                        continue

                assert self.mpd_provider.mpd is not None
                if self.mpd_provider.mpd.type == "dynamic":
                    await self.mpd_provider.update()
                    self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)

                # last_segment = max(self.adaptation_sets[0].representations[0].segments.keys())
                # first_segment = min(self.adaptation_sets[0].representations[0].segments.keys())
                first_segment, last_segment = self.segment_limits(self.adaptation_sets)
                self.log.info(f"{first_segment=}, {last_segment=}")

                if self._index < first_segment:
                    self.log.info(f"Segment {self._index} not in mpd, Moving to next segment")
                    self._index += 1
                    continue

                if self.mpd_provider.mpd.type == "dynamic" and self._index > last_segment:
                    self.log.info(f"Waiting for more segments in mpd : {self.mpd_provider.mpd.type}")
                    await asyncio.sleep(self.time_factor * self.update_interval)
                    continue
                
                if self._index == self._dropped_index:
                    selections = self.abr_controller.update_selection_lowest(self.adaptation_sets)
                else:
                    selections = self.abr_controller.update_selection(self.adaptation_sets, self._index)
                self.log.info(f"Downloading index {self._index} at {selections}")
                self._current_selections = selections

                self.log.info(f'{selections=}')
                # All adaptation sets take the current bandwidth
                adap_bw = {as_id: self.bandwidth_meter.bandwidth for as_id in selections.keys()}

                # Get segments to download for each adaptation set
                try:
                    segments = {
                        adaptation_set_id: self.adaptation_sets[adaptation_set_id].representations[selection].segments[self._index]
                        for adaptation_set_id, selection in selections.items()
                    }
                except KeyError:
                    # No more segments left
                    self.log.info("No more segments left")
                    self._end = True
                    return

                # Download one segment from each adaptation set

                self.log.info(f"Download starting: {selections=}, {segments=}, {adap_bw=}")

                for listener in self.listeners:
                    await listener.on_segment_download_start(self._index, adap_bw, segments)
                
                # Called ~once per segment (once per segment + reinitializations)
                urls = []
                for adaptation_set_id, selection in selections.items():
                    adaptation_set = self.adaptation_sets[adaptation_set_id]
                    representation = adaptation_set.representations[selection]
                    representation_str = "%d:%d" % (adaptation_set_id, representation.id)

                    # Download initial segment
                    if representation_str not in self._representation_initialized:
                        await self.download_manager.download(DownloadRequest(representation.initialization, DownloadType.STREAM_INIT))
                        await self.download_manager.wait_complete(representation.initialization)
                        self.log.info(f"Segment {self._index} Complete. Move to next segment")
                        self._representation_initialized.add(representation_str)

                    # Download next segment
                    try:
                        segment = representation.segments[self._index]
                    except IndexError:
                        self.log.info("Segments ended")
                        self._end = True
                        return
                    urls.append(segment.url)

                    await self.download_manager.download(DownloadRequest(segment.url, DownloadType.SEGMENT))

                self.log.info(f"Waiting for completion urls {urls}")
                results = [await self.download_manager.wait_complete(url) for url in urls]
                self.log.info(f"Completed downloading from urls {urls}")
                if any([result is None for result in results]):
                    # Result is None means the stream got dropped
                    self._dropped_index = self._index
                    continue
                download_stats = {as_id: self.bandwidth_meter.get_stats(segment.url) for as_id, segment in segments.items()}
                for listener in self.listeners:
                    await listener.on_segment_download_complete(self._index, segments, download_stats)
                self._index += 1
                await self.buffer_manager.enqueue_buffer(segments)

    def select_adaptation_sets(self, adaptation_sets: Dict[int, AdaptationSet]):
        as_ids = adaptation_sets.keys()
        start = self.selected_as_start or min(as_ids)
        end = self.selected_as_end or max(as_ids)
        print(f"{start=}, {end=}")
        return {as_id: as_val for as_id, as_val in adaptation_sets.items() if as_id >= start and as_id <= end}

    async def stop(self):
        await self.download_manager.close()
        if self._task is not None:
            self._task.cancel()

    @property
    def is_end(self):
        return self._end

    def add_listener(self, listener: SchedulerEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    async def cancel_task(self, index: int):
        """
        Cancel current downloading task, and move to the next one

        Parameters
        ----------
        index: int
            The index of segment to cancel
        """

        # If the index is the the index of currently downloading segment, ignore it
        if self._index != index or self._current_selections is None:
            return

        # Do not cancel the task for the first index
        if index == 0:
            return

        assert self.adaptation_sets is not None
        for adaptation_set_id, selection in self._current_selections.items():
            segment = self.adaptation_sets[adaptation_set_id].representations[selection].segments[self._index]
            self.log.debug(f"Stop current downloading URL: {segment.url}")
            await self.download_manager.stop(segment.url)

    async def drop_index(self, index):
        self._dropped_index = index

    def switch_score(self, plan):
        switches = list(zip(plan[:-1], plan[1:]))

        s = self.settings
        def diff(switch):
            n,m = switch
            return s.p(s.q(n) - s.q(m))
        
        return map(diff, switches)

    async def score(self, event, plan):
        # Convenience
        s = self.settings
        qualities = [r[1].bandwidth for r in plan]
        switches = self.switch_score(qualities)
        stall_time = 0
        wait_time = 0

        # self.max_buffer_duration
        t = 0
        buffer_level = self.buffer_manager.buffer_level
        for segment in qualities:
            # TODO: segment -> segment.filesize or whatever
            dl_time = event.download_time(segment/(2*1000*1000), t)
            t += dl_time
            buffer_level = buffer_level - dl_time
            if buffer_level < 0:
                stall_time -= buffer_level
                buffer_level = 0
            # TODO: Get segment size correctly through buffer manager or whatever
            buffer_level += 0.5
            if buffer_level > self.max_buffer_duration:
                wait_time += buffer_level - self.max_buffer_duration
                t += buffer_level - self.max_buffer_duration
                buffer_level = self.max_buffer_duration

            if [r[0] for r in plan] == [5,5,5,5,5]:
                self.log.info(f'{plan=}, {qualities=}')
                self.log.info(f'{t=}, {stall_time=}, {buffer_level=}, {wait_time=}, {dl_time=}')


        components = (sum(map(s.q, qualities)), sum(switches), stall_time, wait_time)
        if [r[0] for r in plan] == [5,5,5,5,5]:
            self.log.info(f'{components=}')
            #exit()

        # TODO: Less stupid way of doing this, like make w a numpy vector
        score = s.w1 * components[0] - s.w2 * components[1] - s.w3 * components[2] - s.w4 * components[3] 
        return (score, components, plan)

    async def check_messages(self):
        message = None
        try:
            message = self.receiver.recv_string(flags=zmq.NOBLOCK)
            self.log.info(f'message received, {message=}')
            return message
        except zmq.Again:
            # No message, continue with other tasks
            #self.log.info(f'no message received, continuing')
            return None