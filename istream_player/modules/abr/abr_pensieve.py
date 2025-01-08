import logging

from collections import OrderedDict
from typing import Dict, Optional

from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
from istream_player.core.buffer import BufferManager
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet

import itertools

import torch
import torch.nn as nn
#import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

'''
    WARNING: Bad code ahead. Lasciate ogne speranza, voi ch'intrate.

    A lot of this code is adapted from BONES, which you can find on Github: 
    I use the word "adapted" very loosely as we have totally separate ways of
    actually downloading segments.

    A lot of this code is bad because it needs to be cleaned, it should function
    just fine, but modifying it might be harrowing.
'''

@ModuleOption("pensieve", requires=[BandwidthMeter, BufferManager])
class PensieveABRController(Module, ABRController):
    log = logging.getLogger("DashABRController")

    def __init__(self):
        self.rate_map = None

    async def setup(
        self, config: PlayerConfig, 
        bandwidth_meter: BandwidthMeter, buffer_manager: BufferManager, 
        **kwargs
    ):
        self.k = 8

        self.bandwidth_meter = bandwidth_meter
        self.bandwidth_history = list()

        self.buffer_size = config.buffer_duration
        self.buffer_manager = buffer_manager

        # Per adaptation set btw
        self.bitrate_history = dict()

        # Must be updated externally
        self.download_times = []

        # Initialize the state dict
        self.actor = self.initialize_full_actor_bones()
        ckpt = torch.load(config.pensieve_weights)
        #self.log.info(ckpt['tConv1d.weight'].shape)
        for k,v in ckpt.items():
            self.log.info(f'{k}, {v.shape}')
        self.actor.load_state_dict(torch.load(config.pensieve_weights))
        self.log.info(f'Loaded Pensieve weights from {config.pensieve_weights}')

    def initialize_simple_actor(self):
        '''
            Initialize the simple version of the actor from BONES.
            Unfortunately, everything has to line up right now.
        '''
        return ActorSimple(24, 256, 5, 8)
    
    def initialize_full_actor_bones(self):
        # TODO: Write up BONES actor
        return ActorBetter()

    def initialize_actor(self):
        '''
            Initialize the full version of the actor
        '''
        return Actor()

    def initialize_critic(self):
        return Critic()

    def update_selection(
        self, adaptation_sets: Dict[int, AdaptationSet], index: int
    ) -> Dict[int, int]:
        final_selections = dict()

        # TODO: CHOOSE IDEAL METHOD CALLED HERE
        # Replace with some better way of mapping calls correctly.
        for adaptation_set in adaptation_sets.values():
            final_selections[adaptation_set.id] = (
                self.choose_ideal_simple(adaptation_set)
            )
        
        self.bandwidth_history += [self.bandwidth_meter.bandwidth]

        # Update selection history
        # We don't actually get the adaptation set until here, maybe work this out earlier?
        for adaptation_set_id, selection in final_selections.items():
            if adaptation_set_id in self.bitrate_history.keys():
                self.bitrate_history[adaptation_set_id].append(selection)
            else:
                self.bitrate_history[adaptation_set_id] = [selection]

        return final_selections
    
    def choose_ideal_simple(self, adaptation_set):
        bitrates = [
            representation.bandwidth
            for representation in adaptation_set.representations.values()
        ]

        if adaptation_set.id in self.bitrate_history.keys():
            bitrate_history = self.bitrate_history[adaptation_set.id]
            prev_bitrate = bitrates[bitrate_history[-1]]
        else:
            prev_bitrate = 0

        _, last_segment = self.segment_limits(adaptation_set)
        chunks_remaining = last_segment - len(self.bitrate_history)

        #self.log.info(self.bandwidth_history, self.download_times, bitrates, self.buffer_manager.buffer_level, chunks_remaining, prev_bitrate)
        input = self.actor.parse_input(self.bandwidth_history, 
                                       self.download_times, 
                                       bitrates,
                                       self.buffer_manager.buffer_level, 
                                       chunks_remaining, 
                                       prev_bitrate)

        with torch.no_grad():
            choice = self.actor(input)
            #choice = action_distribution.sample()

        return int(choice)+1

    def choose_ideal_selection_pensieve(self, adaptation_set):
        # Assemble state
        # x_t
        throughputs = self.bandwidth_history[:self.k]
        # Pad
        if len(throughputs) < self.k:
            throughputs = [0] * (self.k - len(throughputs)) + throughputs

        # tau_t 
        download_times = self.download_times[:self.k]
        # Pad
        if len(download_times) < self.k:
            download_times = [0] * (self.k - len(download_times)) + download_times

        # n_t 
        bitrates = [
            representation.bandwidth
            for representation in adaptation_set.representations.values()
        ]
        
        # This probably needs to be scaled somehow
        bitrates.sort()

        # b_t
        buffer_level = self.buffer_manager.buffer_level

        # c_t
        first_segment, last_segment = self.segment_limits(adaptation_set)
        chunks_remaining = last_segment - len(self.bitrate_history)

        # l_t
        if adaptation_set.id in self.bitrate_history.keys():
            bitrate_history = self.bitrate_history[adaptation_set.id]
            prev_bitrate = bitrates[bitrate_history[-1]]
        else:
            prev_bitrate = 0
        
        throughputs_t = torch.tensor([[throughputs]], 
                                     dtype=torch.float32)
        download_times_t = torch.tensor([[download_times]], 
                                        dtype=torch.float32)
        bitrates_t = torch.tensor([[bitrates]], 
                                  dtype=torch.float32)
        scalars = torch.tensor([buffer_level, 
                                chunks_remaining, 
                                prev_bitrate], 
                                dtype=torch.float32)

        with torch.no_grad():
            choice = self.actor(throughputs_t, download_times_t, bitrates_t, scalars)
            choice = int(torch.argmax(choice))

        return choice

    def segment_limits(self, adaptation_set: Dict[int, AdaptationSet]) -> tuple[int, int]:
        ids = [[seg_id for seg_id in repr.segments.keys()] for repr in adaptation_set.representations.values()]
        ids = list(itertools.chain(*ids))

        return min(ids), max(ids)
    
    def update_download_time(self, download_time):
        ''' Add download_time to internal download time counter. '''
        self.download_times += [download_time]

class Actor(nn.Module):
    def __init__(self):
        # Fully configurable
        history_size = 8
        out_channels = 128
        # Must be smaller than history_size
        kernel_size = 4
        # Must match number of bitrates
        num_actions = 6
        # Cannot be changed without changing arch massively
        input_channels = 3

        super(Actor, self).__init__()

        # Parameters need adjustment?
        self.throughput_history = nn.Conv1d(1, out_channels, kernel_size, 1)
        self.download_time_history = nn.Conv1d(1, out_channels, kernel_size, 1)
        self.next_bitrate = nn.Conv1d(1, out_channels, kernel_size, 1)

        fcn_inputs = ((history_size - (kernel_size-1))*out_channels)*2 + (num_actions - (kernel_size-1))*out_channels
        self.conv_fcn = nn.Linear(fcn_inputs, 128)

        self.scalar_inputs = nn.Linear(3, 128)
        self.combine_step = nn.Linear(256, num_actions)

    def forward(self, throughput_history, download_time_history, next_bitrates, scalars):
        # Convolutional portion
        throughput_conv = self.throughput_history(throughput_history)
        throughput_conv = F.relu(throughput_conv)
        throughput_conv = torch.flatten(throughput_conv)

        download_time_conv = self.download_time_history(download_time_history)
        download_time_conv = F.relu(download_time_conv)
        download_time_conv = torch.flatten(download_time_conv)

        next_bitrate_conv = self.next_bitrate(next_bitrates)
        next_bitrate_conv = F.relu(next_bitrate_conv)
        next_bitrate_conv = torch.flatten(next_bitrate_conv)

        fcn_inputs = torch.cat([throughput_conv, download_time_conv, next_bitrate_conv])
        fcn_inputs = torch.flatten(fcn_inputs)
        conv_fcn = self.conv_fcn(fcn_inputs)

        # Fully connected portion
        scalar_fcn = self.scalar_inputs(scalars)
        scalar_fcn = F.relu(scalar_fcn)

        # Combine the two inputs
        combine_input = torch.cat([conv_fcn, scalar_fcn])
        combine_input = torch.flatten(combine_input)
        combine_input = self.combine_step(combine_input)
        combine_input = F.relu(combine_input)

        output = F.log_softmax(combine_input, dim=-1)

        return output

class Critic(nn.Module):
    def __init__(self):
        input_channels = 3
        num_actions = 6
        out_channels = 32
        kernel_size = 3

        super(Actor, self).__init__()

        # Parameters need adjustment?
        self.conv_inputs = nn.Conv1d(input_channels, out_channels, kernel_size, 1)
        self.scalar_inputs = nn.Linear((num_actions - (kernel_size-1))*out_channels, 128)
        self.combine_step = nn.Linear(128, num_actions)

    def forward(self, x, tau, n, b, c, l):
        # Probably do this differently? Is this true?
        x_bar = np.concat(x, tau, n)
        x_bar = self.conv_inputs(x_bar)
        x_bar = F.relu(x_bar)
        x_bar = F.max_pool2d(x_bar, 2)
        x_bar = torch.flatten(x_bar, 1)

        scalars = np.concat(b, c, l)
        scalars = self.scalar_inputs(scalars)
        scalars = F.relu(scalars)

        combine_input = np.concat(x_bar, scalars)
        combine_input = self.combine_step(combine_input)

        output = F.relu(combine_input)

        return output

class ActorSimple(nn.Module):
    ''' A simpler version of the model taken from the BONES paper. '''
    def __init__(self, state_dim, hidden_units, action_dim, k):
        super(ActorSimple, self).__init__()

        # Input history length
        self.k = k

        self.fc1 = nn.Linear(state_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        distribution = Categorical(F.softmax(self.fc2(x)))
        choice = distribution.sample()
        return choice
    
    def parse_input(self, 
                    throughput_history, 
                    download_time_history, 
                    next_bitrates, 
                    buffer_level, 
                    chunks_remaining, 
                    prev_bitrate):
        # Asseble the state as is to into a usable vector
        action_dim = self.fc2.out_features

        # x_t
        throughputs = throughput_history[:self.k]
        # Pad
        if len(throughputs) < self.k:
            throughputs = [0] * (self.k - len(throughputs)) + throughputs

        # tau_t 
        download_times = download_time_history[:self.k]
        # Pad
        if len(download_times) < self.k:
            download_times = [0] * (self.k - len(download_times)) + download_times

        # n_t 
        bitrates = sorted(next_bitrates)[-action_dim:]

        throughputs_t = torch.tensor(throughputs, dtype=torch.float32) / 8000
        download_times_t = torch.tensor(download_times, dtype=torch.float32)
        bitrates_t = torch.tensor(bitrates, dtype=torch.float32) / np.max(bitrates)
        # l_t
        out = torch.tensor([buffer_level, prev_bitrate, chunks_remaining], dtype=torch.float32)

        return torch.cat((out, throughputs_t, download_times_t, bitrates_t))

class ActorBetter(nn.Module):
    # This should be parameterized but if we're loading the model there's kinda no point to doing that no?
    #def __init__(self, input_dim, hidden_dim, output_dim, kernel_size):
    def __init__(self):
        super(ActorBetter, self).__init__()

        self.k = 8
        
        # 1D Convolution layers
        self.tConv1d = nn.Conv1d(1, 128, kernel_size=4)
        self.dConv1d = nn.Conv1d(1, 128, kernel_size=4)
        self.cConv1d = nn.Conv1d(1, 128, kernel_size=4)
        
        # Fully connected layers for different inputs
        self.bufferFc = nn.Linear(1, 128)
        self.leftChunkFc = nn.Linear(1, 128)
        self.bitrateFc = nn.Linear(1, 128)
        
        # Main fully connected layers
        self.fullyConnected = nn.Linear(15 * 128, 128)  # 1920 = 15 * 128
        self.outputLayer = nn.Linear(128, 5)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Assuming x contains all necessary inputs in appropriate format
        t_conv = self.relu(self.tConv1d(x['throughput']))
        d_conv = self.relu(self.dConv1d(x['download']))
        c_conv = self.relu(self.cConv1d(x['chunk']))
        
        buffer_fc = self.relu(self.bufferFc(x['buffer']))
        left_chunk_fc = self.relu(self.leftChunkFc(x['left_chunk']))
        bitrate_fc = self.relu(self.bitrateFc(x['bitrate']))
        
        # Flatten and concatenate all features
        conv_flat = torch.cat([t_conv.flatten(1), d_conv.flatten(1), c_conv.flatten(1)], dim=1)
        fc_flat = torch.cat([buffer_fc, left_chunk_fc, bitrate_fc], dim=1)
        combined = torch.cat([conv_flat, fc_flat], dim=1)
        
        # Final layers
        hidden = self.relu(self.fullyConnected(combined))
        logits = self.outputLayer(hidden)
        
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
        
        return action
        #, probs
        
        #return output

    def parse_input(self, 
                throughput_history, 
                download_time_history, 
                next_bitrates, 
                buffer_level, 
                chunks_remaining, 
                prev_bitrate):
        # Prepare sequences for Conv1d (shape: [1, 1, k])
        throughputs = throughput_history[:self.k]
        if len(throughputs) < self.k:
            throughputs = [0] * (self.k - len(throughputs)) + throughputs
        throughputs_t = torch.tensor(throughputs, dtype=torch.float32).view(1, 1, -1) / 8000

        downloads = download_time_history[:self.k]
        if len(downloads) < self.k:
            downloads = [0] * (self.k - len(downloads)) + downloads
        downloads_t = torch.tensor(downloads, dtype=torch.float32).view(1, 1, -1)

        # Prepare chunk sizes (assuming they correlate with bitrates)
        bitrates = sorted(next_bitrates)[-5:]
        chunks_t = torch.tensor(bitrates, dtype=torch.float32).view(1, 1, -1) / np.max(bitrates)

        # Scalar inputs
        buffer_t = torch.tensor([[buffer_level]], dtype=torch.float32)
        left_chunk_t = torch.tensor([[chunks_remaining]], dtype=torch.float32)
        bitrate_t = torch.tensor([[prev_bitrate]], dtype=torch.float32)

        return {
            'throughput': throughputs_t,
            'download': downloads_t,
            'chunk': chunks_t,
            'buffer': buffer_t,
            'left_chunk': left_chunk_t,
            'bitrate': bitrate_t
        }