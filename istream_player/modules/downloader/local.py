import asyncio
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from istream_player.config.config import PlayerConfig
from istream_player.core.downloader import (DownloadEventListener,
                                            DownloadManager, DownloadRequest)
from istream_player.core.module import Module, ModuleOption


@ModuleOption("local", default=True)
class LocalClient(Module, DownloadManager):
    # Controls the bandwidth of the local downloader in bps
    #def __init__(self, *, bw="100_000_000_000") -> None:
    def __init__(self, *, bw="1_000_000") -> None:
        # 1_333_000_000
        super().__init__()
        self.bw = int(bw)  # Not implemented
        self.max_packet_size = 20_000

        self.transfer_queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()
        self.content: Dict[str, bytearray] = defaultdict(bytearray)
        self.transfer_size: Dict[str, int] = {}
        self.transfer_compl: Dict[str, asyncio.Event] = {}
        self.downloader_task: Optional[asyncio.Task] = None

    async def setup(self, config: PlayerConfig, **kwargs):
        try:
            bw_history = csv_to_dict(config.bandwidth_history_trace_file)
        except:
            bw_history = None
        self.downloader_task = asyncio.create_task(self.throttled_download(bw_history), name="TASK_LOCAL_DOWNLOADER")
        self.time_factor = config.time_factor

    async def cleanup(self):
        if self.downloader_task:
            self.downloader_task.cancel()

    async def wait_complete(self, url: str) -> Tuple[bytes, int]:
        """
        Wait the stream to complete

        Parameters
        ----------
        url:
            The URL to wait for

        Returns
        -------
            The return value could be None, meaning that the stream got dropped.
            It could be a tuple, the bytes as the first element and size as the second element.
        """
        await self.transfer_compl[url].wait()
        content = self.content[url]
        del self.content[url]
        del self.transfer_compl[url]
        del self.transfer_size[url]
        return content, len(content)

    def cancel_read_url(self, url: str):
        raise Exception("Local Downloader : Cannot cancel download")

    async def drop_url(self, url: str):
        """
        Drop the URL downloading process
        """
        raise Exception("Local Downloader : Cannot drop download")

    @property
    def is_busy(self):
        return False

    async def download(self, request: DownloadRequest, save: bool = False) -> Optional[bytes]:
        url = request.url
        self.transfer_compl[url] = asyncio.Event()
        self.transfer_size[url] = Path(url).stat().st_size
        for listener in self.listeners:
            await listener.on_transfer_start(url)
        asyncio.create_task(self.request_read(url), name=f"TASK_LOCAL_REQREAD_{url.rsplit('/', 1)[-1]}")
        if save:
            await self.transfer_compl[url].wait()
            content = self.content[url]
            return content
        else:
            return None

    async def close(self):
        pass

    async def stop(self, url: str):
        pass

    def add_listener(self, listener: DownloadEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    async def request_read(self, url: str):
        # print(f"Request : {url}")
        with open(url, "rb") as f:
            while True:
                data = f.read(self.max_packet_size)
                # print(f"Putting {len(data)} bytes for {url}")
                await self.transfer_queue.put((url, data))
                if not data:
                    break

    async def throttled_download(self, bw_history: Dict[float, int]):
        '''
            bw_history: Maps time in seconds to a bandwidth in bps
        '''
        # Time in seconds
        time = 0
        if bw_history is not None:
            keys_by_index = list(bw_history.keys())
        while True:
            print("Getting response from transfer_queue")
            url, chunk = await self.transfer_queue.get()
            if chunk:
                self.content[url].extend(chunk)
                for listener in self.listeners:
                    await listener.on_bytes_transferred(
                        len(chunk), url, len(self.content[url]), self.transfer_size[url], chunk
                    )
            else:
                self.transfer_compl[url].set()
                for listener in self.listeners:
                    await listener.on_transfer_end(self.transfer_size[url], url)
            # Add bandwidth limiting here
            time += float(self.time_factor) * float(self.max_packet_size) / float(self.bw)
            print(time)
            if bw_history is not None:
                if time < float(keys_by_index[1]):
                    print('0 time case')
                    self.bw = bw_history[keys_by_index[0]]
                    print(self.bw)
                else:
                    print('time > 1')
                    for i in range(1, len(keys_by_index)):
                        print('searching for key: ', i)
                        if float(keys_by_index[i]) > time:
                            print('found key: ', i)
                            self.bw = bw_history[keys_by_index[i-1]]
                            print("BW: ", self.bw)
                            break
                        print('broke?: ', i)
            # else bandwidth constant
            print("After discovering bandwidth:", float(self.time_factor) * float(self.max_packet_size) / float(self.bw))
            await asyncio.sleep(float(self.time_factor) * float(self.max_packet_size) / float(self.bw))

def csv_to_dict(csv_file):
    '''
        Read csv into dictionary
    '''
    if csv_file == None:
        return None
    result_dict = {}
    with open(csv_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            result_dict[row[0]] = row[1]
    return result_dict