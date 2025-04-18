import sys
import math
import yaml
import random
import os
import pathloss_3gpp_eq24
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from collections import deque
from ts_eq24 import Antenna, Packet, UE


def slot_traffic_creation(data_case, antennas, ues, current_time, tick):
    from collections import deque

    # === Parameters from the scenario configuration ===
    NOH = data_case["ETUDE_DE_TRANSMISSION"]["OVERHEAD"]["Bits"]  # Overhead bits per slot
    NRE_PER_RB = 12 * 14 - NOH  # Resource Elements per RB in one slot (12 subcarriers * 14 OFDM symbols)
    BITS_PER_RE = 1  # Modulation (e.g., QPSK = 1 bit per RE, 16QAM = 4 bits per RE, etc.)
    all_priorities = {"app3": 1, "app2": 2, "app1": 3}  # Application priorities: Auto > Drone > Streaming

    # === Prepare UE packet queues for efficient popping ===
    for ue in ues:
        if not isinstance(ue.arrivals, deque):
            ue.arrivals = deque(ue.arrivals)
        if not isinstance(ue.packets, deque):
            ue.packets = deque(ue.packets)

    # === Loop through all antennas ===
    for antenna in antennas:
        if antenna.nrb is None or antenna.nrb == 0:
            continue
        current_slot_packets = []  # This will hold the packets actually transmitted in this slot
        slot_rb_total = antenna.nrb  # Number of available Resource Blocks in this slot
        rb_index = 0  # Index of the next free RB
        all_ready_packets = []  # Will contain (priority, arrival, eff_rb_bits, pkt_size, ue, pkt)

        # === For each UE associated with this antenna, collect packets ready to be sent ===
        for ue in ues:
            if ue.assoc_ant == antenna.id:
                app = ue.app.lower()
                priority = all_priorities.get(app, 99)
                eff = ue.eff  # Spectral efficiency
                eff_rb_bits = eff * NRE_PER_RB * BITS_PER_RE  # Number of bits that 1 RB can carry

                # Fetch packets that are ready to be sent at current_time
                while ue.arrivals and current_time >= ue.arrivals[0]:
                    pkt_size = ue.packets.popleft()
                    ue.arrivals.popleft()
                    pkt = Packet(
                        source=ue,
                        app=ue.app,
                        packet_id=len(ue.packets),
                        packet_size=pkt_size,
                        timeTX=current_time
                    )
                    all_ready_packets.append(
                        (priority, current_time, eff_rb_bits, pkt_size, ue, pkt)
                    )

        # === Sort all packets by priority (low = higher priority), then by arrival time ===
        all_ready_packets.sort(key=lambda x: (x[0], x[1]))

        # === Schedule packets into this slot ===
        for _, _, eff_rb_bits, pkt_size, ue, pkt in all_ready_packets:
            if eff_rb_bits <= 0:
                continue  # Cannot schedule if spectral efficiency is invalid

            remaining_rb = slot_rb_total - rb_index
            rb_needed = math.ceil(pkt_size / eff_rb_bits)

            if rb_needed <= remaining_rb:
                # The whole packet fits in remaining RBs → schedule it fully
                current_slot_packets.append(pkt)
                rb_index += rb_needed

            else:
                # Only a fragment of the packet can fit in this slot
                fragment_bits = int(remaining_rb * eff_rb_bits)
                if fragment_bits >= 1:
                    # Create and schedule fragment
                    fragment_pkt = Packet(
                        source=pkt.source,
                        app=pkt.app,
                        packet_id=pkt.id,
                        packet_size=fragment_bits,
                        timeTX=current_time
                    )
                    current_slot_packets.append(fragment_pkt)

                    # Requeue the rest of the packet at the front of UE's queue
                    ue.arrivals.appendleft(pkt.timeTX)
                    ue.packets.appendleft(pkt_size - fragment_bits)
                    rb_index += remaining_rb

                else:
                    # Not even a fragment fits → defer whole packet
                    ue.arrivals.appendleft(pkt.timeTX)
                    ue.packets.appendleft(pkt_size)

                break  # Stop trying to add packets, the slot is full

        # === Store the result for this slot ===
        antenna.packet_queues_slot.append(current_slot_packets)

        # Append to tick-wide packet queue (for statistics)
        if tick == len(antenna.packet_queues_tick) - 1:
            antenna.packet_queues_tick[-1].extend(current_slot_packets)
        else:
            antenna.packet_queues_tick.append(current_slot_packets)
