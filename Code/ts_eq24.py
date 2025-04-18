## Écrire votre numéro d'équipe
## Érire les noms et matricules de chaque membre de l'équipe
## CECI EST OBLIGATOIRE

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
from traffic import *


##############################################
#                 CLASSES                    #
##############################################

class Antenna:
    def __init__(self, id):
        self.id = id
        self.frequency = None
        self.height = None
        self.group = None
        self.coords = None
        self.assoc_ues = []
        self.scenario = None
        self.gen = None
        self.packet_queues_slot = []        
        self.packet_queues_tick = []
        self.all_packets = []
        self.current_slot = None
        self.packets_this_slot = 0
        self.bits_this_slot = 0
        self.nrb = None

    def __repr__(self):
        return f"Antenna(id={self.id}, freq={self.frequency}GHz, RBs={self.nrb}, UEs={len(self.assoc_ues)})"

class Packet:
    def __init__(self, source, app, packet_id, packet_size, timeTX):
        self.id = packet_id
        self.size = packet_size
        self.timeTX = timeTX
        self.timeRX = None
        self.app = app
        self.source = source

    def __repr__(self):
        return (f"size={self.size}, timeTX={self.timeTX}, app={self.app}, source={self.source}")


class UE:
    def __init__(self, id, app_name):
        self.id = id
        self.height = None
        self.group = None
        self.coords = None
        self.app = app_name
        self.assoc_ant = None
        self.los = True
        self.gen = None
        self.packets = []
        self.assoc_ant_pl = None
        self.cqi = None
        self.eff = None
        self.nre = None
        self.ninfo = None
        self.packets = []
        self.arrivals = []

    def __repr__(self):
        return f"UE(id={self.id}, app={self.app}, coords={self.coords}, ant={self.assoc_ant}, cqi={self.cqi})"

##############################################
#       TRAFFIC MANAGEMENT FUNCTIONS         #
##############################################

def generate_expo_inter_arrivals(tfinal, inter_mean_ms):
    inter_mean_s = inter_mean_ms / 1000.0
    inter_arrivals = []
    total_time = 0.0
    while total_time < tfinal:
        interval = (random.expovariate(1.0 / inter_mean_s))*1000        
        total_time += interval
        if total_time <= tfinal:
            inter_arrivals.append(interval)
        else:
            break
    return inter_arrivals

def generate_uniform_inter_arrivals(tfinal, min_ms, max_ms):
    min_s = min_ms / 1000.0
    max_s = max_ms / 1000.0
    inter_arrivals = []
    total_time = 0.0
    while total_time < tfinal:
        interval = (random.uniform(min_s, max_s))*1000
        total_time += interval
        if total_time <= tfinal:
            inter_arrivals.append(interval)
        else:
            break
    return inter_arrivals

def generate_packet_length_and_arrivals(data_case, devices, ues):
    tfinal = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tfinal"]

    for ue in ues:
        if ue.app.lower() == "app1":
            inter_mean = devices["UES"]["UE1-App1"]["inter_mean_ms"]
            base_bits = devices["UES"]["UE1-App1"]["base_bits"]
            variability = devices["UES"]["UE1-App1"]["variability"]
            inter_arrivals = generate_expo_inter_arrivals(tfinal, inter_mean)

        elif ue.app.lower() == "app2":
            inter_min = devices["UES"]["UE2-App2"]["inter_min_ms"]
            inter_max = devices["UES"]["UE2-App2"]["inter_max_ms"]
            base_bits = devices["UES"]["UE2-App2"]["base_bits"]
            variability = devices["UES"]["UE2-App2"]["variability"]
            inter_arrivals = generate_uniform_inter_arrivals(tfinal, inter_min, inter_max)

        elif ue.app.lower() == "app3":
            inter_min = devices["UES"]["UE3-App3"]["inter_min_ms"]
            inter_max = devices["UES"]["UE3-App3"]["inter_max_ms"]
            base_bits = devices["UES"]["UE3-App3"]["base_bits"]
            variability = devices["UES"]["UE3-App3"]["variability"]
            inter_arrivals = generate_uniform_inter_arrivals(tfinal, inter_min, inter_max)

        else:
            raise ValueError(f"Type d'application inconnu: {ue.app}")

        ue.arrivals = list(np.cumsum(inter_arrivals))
        ue.packets = [
            int(random.uniform(base_bits * (1 - variability),
                               base_bits * (1 + variability)))
            for _ in range(len(ue.arrivals))
        ]

        print(f"\rUE traffic: {ue.id}", end="")

def plot_transmission_summary(packet_counts_per_tick):
    ticks = set()
    packet_counts = defaultdict(int)
    byte_counts = defaultdict(int)

    for entry in packet_counts_per_tick:
        tick = entry['tick']
        pkt_count = entry['packet_count']
        byte_count = entry['total_bytes']

        packet_counts[tick] += pkt_count
        byte_counts[tick] += byte_count

    # Create ordered lists
    ticks = sorted(packet_counts.keys())
    total_pkts_per_tick = [packet_counts[tick] for tick in ticks]
    total_bytes_per_tick = [byte_counts[tick] for tick in ticks]
    avg_pkt_size_per_tick = [
        byte_counts[t] / packet_counts[t] if packet_counts[t] else 0 for t in ticks
    ]

    plt.figure(figsize=(8, 4))
    plt.hist(total_pkts_per_tick, bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram: Packets Transmitted per Tick")
    plt.xlabel("Packets per Tick")
    plt.ylabel("Number of Ticks")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(total_bytes_per_tick, bins=20, color='orange', edgecolor='black')
    plt.title("Histogram: Bytes Transmitted per Tick")
    plt.xlabel("Bytes per Tick")
    plt.ylabel("Number of Ticks")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(avg_pkt_size_per_tick, bins=20, color='green', edgecolor='black')
    plt.title("Histogram: Average Packet Size per Tick")
    plt.xlabel("Average Packet Size (Bytes)")
    plt.ylabel("Number of Ticks")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



##############################################
#       RESOURCE ALLOCATION FUNCTIONS       #
##############################################

def get_nrb_from_bw_scs(bw_mhz, scs_khz):
    table = {
        15: [(4.32, 24), (49.5, 275)],
        30: [(8.64, 24), (99, 275)],
        60: [(17.28, 24), (198, 275)],
        120: [(34.56, 24), (396, 275)],
        240: [(69.12, 24), (397.44, 138)]
    }
    for scs, limits in table.items():
        if scs_khz == scs:
            min_bw, min_nrb = limits[0]
            max_bw, max_nrb = limits[1]
            if bw_mhz <= min_bw:
                return min_nrb
            elif bw_mhz >= max_bw:
                return max_nrb
            else:
                ratio = (bw_mhz - min_bw) / (max_bw - min_bw)
                return round(min_nrb + ratio * (max_nrb - min_nrb))
    return 0

def compute_antenna_load_weights(antennas, ues):
    group_weights = {
        'UE1-App1': 2000000,
        'UE2-App2': 2857,
        'UE3-App3': 100
    }
    antenna_weights = {}
    for antenna in antennas:
        total = 0
        for ue_id in antenna.assoc_ues:
            ue = next(u for u in ues if u.id == ue_id)
            group = ue.group
            poids = group_weights.get(group, 0)
            total += poids
        antenna_weights[antenna.id] = total
    return antenna_weights

def assign_rb_proportionally(total_nrb, antenna_weights, antennas):
    total_weight = sum(antenna_weights.values())
    nrb_alloc = {}
    used_rb = 0

    # Step 1: Give 1 RB to all antennas with weight > 0
    for antenna in antennas:
        weight = antenna_weights.get(antenna.id, 0)
        if weight > 0:
            nrb_alloc[antenna.id] = 1
            used_rb += 1
        else:
            nrb_alloc[antenna.id] = 0  # No UEs or no weight

    # Step 2: Distribute the remaining RBs proportionally
    remaining_rb = total_nrb - used_rb
    if remaining_rb < 0:
        raise ValueError("Total RBs too small to allocate at least 1 per antenna with UEs.")

    # Get IDs of antennas with weight > 0
    weighted_antennas = [ant for ant in antennas if antenna_weights.get(ant.id, 0) > 0]
    remaining_weight = sum(antenna_weights[ant.id] for ant in weighted_antennas)

    for antenna in weighted_antennas:
        weight = antenna_weights[antenna.id]
        share = (weight / remaining_weight) if remaining_weight > 0 else 0
        extra_rb = int(share * remaining_rb)
        nrb_alloc[antenna.id] += extra_rb
        used_rb += extra_rb

    # Step 3: Distribute any leftover RBs one by one to top-weight antennas
    leftover = total_nrb - used_rb
    sorted_ids = sorted(weighted_antennas, key=lambda a: antenna_weights[a.id], reverse=True)
    for i in range(leftover):
        nrb_alloc[sorted_ids[i % len(sorted_ids)].id] += 1

    # Step 4: Assign final values
    for antenna in antennas:
        antenna.nrb = nrb_alloc[antenna.id]


##############################################
#         PATHLOSS COMPUTATION FUNCTIONS     #
##############################################

def findMinMaxPathloss(plFileName):
    min_val = math.inf
    max_val = -math.inf
    with open(plFileName, "r") as f:
        for line in f:
            parts = line.strip().split()
            value = float(parts[2])
            if(value != math.inf and value != 0): 
                min_val = min(min_val, value)
                max_val = max(max_val, value)
    return (min_val, max_val)

def pathloss_to_cqi(pathloss, minPl, maxPl):
    num_bins = 15
    step = (maxPl - minPl) / num_bins
    if pathloss < minPl:
        return num_bins
    if pathloss >= maxPl:
        return 0
    index = int((pathloss - minPl) / step)
    return num_bins - index

def define_los(filename, ue_id, antenna_id):
    ue_id = int(ue_id)
    antenna_id = int(antenna_id)
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.split()
                if int(parts[0]) == ue_id and antenna_id in map(int, parts[1:]):
                    return False
    return True

def get_efficiency_from_cqi(cqi):

    """
    Get spectral efficiency from CQI based on 3GPP TS 38.214 Table 5.2.2.1-2
    
    (to be added to the report: this table was chosen over the other tables because it's the 
    standard table used in 5G NR implementations;
    It provides a good balance between robustness and throughput;
    Using a single table will make it easier to compare performance across different applications)

    Args:
        cqi: Channel Quality Indicator (0-15)
        
    Returns:
        Spectral efficiency
    """
    # CQI to efficiency mapping based on Table 5.2.2.1-2
    efficiency_table = {
        0: 0.0,    # Out of range - no transmission
        1: 0.1523, # QPSK, code rate 78/1024
        2: 0.2344, # QPSK, code rate 120/1024
        3: 0.3770, # QPSK, code rate 193/1024
        4: 0.6016, # QPSK, code rate 308/1024
        5: 0.8770, # QPSK, code rate 449/1024
        6: 1.1758, # QPSK, code rate 602/1024
        7: 1.4766, # 16QAM, code rate 378/1024
        8: 1.9141, # 16QAM, code rate 490/1024
        9: 2.4063, # 16QAM, code rate 616/1024
        10: 2.7305, # 64QAM, code rate 466/1024
        11: 3.3223, # 64QAM, code rate 567/1024
        12: 3.9023, # 64QAM, code rate 666/1024
        13: 4.5234, # 64QAM, code rate 772/1024
        14: 5.1152, # 64QAM, code rate 873/1024
        15: 5.5547  # 64QAM, code rate 948/1024
    }
    
    # Return the efficiency for the given CQI, or 0.0 if CQI is invalid
    return efficiency_table.get(cqi, 0.0)

def okumura(scenario, frequency, distance, antenna_height, UE_height):
    if distance < 1:
        return 0
    if distance > 20:
        return math.inf
    if scenario == "urban_small":
        correction_factor = 0
    elif scenario == "urban_large":
        correction_factor = -(-(1.1 * math.log10(frequency) - 0.7) * UE_height + (1.56 * math.log10(frequency) - 0.8))
        if frequency <= 300:
            correction_factor -= 8.29*(math.log10(1.54*UE_height))**2 -1.1
        else:
            correction_factor -= 3.2*(math.log10(11.75*UE_height))**2 -4.97
    elif scenario == "suburban":
        correction_factor = -2 * (math.log10(frequency / 28))**2 - 5.4
    elif scenario == "open":
        correction_factor = -4.78 * (math.log10(frequency))**2 + 18.33 * math.log10(frequency) - 40.94
    else:
        raise ValueError("Scénario inconnu. Choisissez parmi: urban_large, urban_small, suburban, open.")
    A = 69.55 + 26.16 * math.log10(frequency) - 13.82 * math.log10(antenna_height)
    B = (44.9 - 6.55 * math.log10(antenna_height)) * math.log10(distance)
    C = correction_factor
    pathloss = A + B + C - (1.1 * math.log10(frequency) - 0.7) * UE_height + (1.56 * math.log10(frequency) - 0.8)
    return pathloss

def threegpp(scenario, frequency, distance, antenna_height, UE_height, ue_id, antenna_id, visibility_file_name):
    los = define_los(visibility_file_name, ue_id, antenna_id)
    if scenario == "RMa":
        return pathloss_3gpp_eq24.rma_los(frequency, distance, antenna_height, UE_height) if los else pathloss_3gpp_eq24.rma_nlos(frequency, distance, antenna_height, UE_height)
    elif scenario == "UMa":
        return pathloss_3gpp_eq24.uma_los(frequency, distance, antenna_height, UE_height) if los else pathloss_3gpp_eq24.uma_nlos(frequency, distance, antenna_height, UE_height)
    elif scenario == "UMi":
        return pathloss_3gpp_eq24.umi_los(frequency, distance, antenna_height, UE_height) if los else pathloss_3gpp_eq24.umi_nlos(frequency, distance, antenna_height, UE_height)
    else:
        raise ValueError(f"Scenario name {scenario} is not available (RMa, UMa or UMi)")
    
def generate_pathlosses(data_case,ues,antennas):
    num_ues = len(ues)
    num_antennas = len(antennas)
    pathlosses = [[None for _ in range(num_antennas)] for _ in range(num_ues)]

    model = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["model"]
    scenario = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["scenario"]

    for ue in ues:
        for antenna in antennas:
            antenna_height = antenna.height
            UE_height = ue.height
            frequency = (antenna.frequency) # GHz
            # Calculer la distance entre l'UE et l'antenne
            distance = math.sqrt((float(ue.coords[0]) - float(antenna.coords[0]))**2 + (float(ue.coords[1]) - float(antenna.coords[1]))**2) / 1000  # en km
            if(model == "okumura"):
                frequency = (antenna.frequency)*1000 # to MHz
                pathlosses[int(ue.id)][int(antenna.id)]=okumura(scenario, frequency, distance, antenna_height, UE_height)
            elif(model == "3gpp"):
                distance = distance * 1000 # to m
                visibility_file_name = data_case["ETUDE_DE_TRANSMISSION"]["VISIBILITY"]["read"]
                pathlosses[int(ue.id)][int(antenna.id)]=threegpp(scenario, frequency, distance, antenna_height, UE_height, ue.id, antenna.id, visibility_file_name)
    return pathlosses

##############################################
#             FILE FUNCTIONS              #
##############################################

def get_from_dict(key, data, res=None, curr_level=1, min_level=1):
    if res:
        return res
    if type(data) is not dict:
        msg = f"get_from_dict() works with dicts and is receiving a {type(data)}"
        ERROR(msg, 1)
    else:
        for k, v in data.items():
            if k == key and curr_level >= min_level:
                return data[k]
            if type(v) is dict:
                level = curr_level + 1
                res = get_from_dict(key, v, res, level, min_level)
    return res

def read_yaml_file(fname):
    with open(fname,'r') as file:
        return yaml.safe_load(file)

def read_segment_file(fname):
    segments = []
    with open(fname,'r') as file:
        for line in file:
            elements = line.strip().split()
            segments.append([int(elements[0]),float(elements[1]),float(elements[2])])
    return segments

def ERROR(msg , code = 1):
    print("\n\n\nERROR\nPROGRAM STOPPED!!!\n")
    if msg:
        print(msg)
    print(f"\n\texit code = {code}\n\n\t\n")
    sys.exit(code)

# Fonction pour créer le fichier de pathloss ts_eq24_pl.txt
def create_pathloss_file(data_case, ues, antennas, pathlosses):
    model = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["model"]
    scenario = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["scenario"]
    with open("ts_eq24_pl.txt", "w") as file:
        for ue in ues:
            for antenna in antennas:
                file.write(f"{ue.id:<5} {antenna.id:<5} {pathlosses[int(ue.id)][int(antenna.id)]:<20} {model:<8} {scenario}\n")

# Fonction pour créer le fichier d'association d'antennes
def create_assoc_files(data_case, ues, antennas,pathlosses):
    # Initialiser les associations
    for antenna in antennas:
        antenna.assoc_ues.clear()

    # Calculer le pathloss pour chaque UE et antenne et associer les UEs aux antennes
    for ue in ues:
        best_antenna = None
        best_pathloss = float('inf')
        for antenna in antennas:
            # Calculer la distance entre l'UE et l'antenne
            pathloss = pathlosses[int(ue.id)][int(antenna.id)]
            if pathloss < best_pathloss:
                best_pathloss = pathloss
                best_antenna = antenna
        
        # Ajouter l'UE à l'antenne avec le pathloss minimal
        if best_antenna:
            best_antenna.assoc_ues.append(ue.id)
            ue.assoc_ant = best_antenna.id

    # Créer le fichier d'association des antennes
    with open("ts_eq24_assoc_ant.txt", "w") as file:
        for antenna in antennas:
            assoc_ues_str = " ".join([f"{ue_id:<5}" for ue_id in antenna.assoc_ues])
            file.write(f"{antenna.id:<3} {assoc_ues_str}\n")
    # Créer le fichier d'association des UEs aux antennes
    with open("ts_eq24_assoc_ue.txt", "w") as file:
        for ue in ues:
            file.write(f"{ue.id:<3} {ue.assoc_ant}\n")

def create_text_file(coord_file_name,antennas,ues):
    with open(coord_file_name,"w") as file:
        for antenna_index in range(len(antennas)):
            file.write(f"antenna {antenna_index}\t")
            file.write(f"{antennas[antenna_index].group}\t")
            file.write(f"{antennas[antenna_index].coords[0]:.1f}\t")
            file.write(f"{antennas[antenna_index].coords[1]:.1f}\t\n")
        for ue_index in range(len(ues)):
            file.write(f"ue     \t{ue_index}\t")
            file.write(f"{ues[ue_index].group}\t")
            file.write(f"{ues[ue_index].coords[0]:.1f}\t")
            file.write(f"{ues[ue_index].coords[1]:.1f}\t")
            file.write(f"{ues[ue_index].app}\n")

def read_coord_file(data_case,devices):

    fname = data_case["ETUDE_DE_TRANSMISSION"]["COORD_FILES"]["read"]

    if(os.path.isfile(os.path.join(os.getcwd(),fname))):
        pass
    else:
        print(f"No coordinates file: \"{fname}\" found")
        exit()

    antennas = []
    ues = []

    with open(fname,'r') as file:
        for line in file:
            elements = line.strip().split()
            if(elements[0] == "antenna"):
                antenna = Antenna(elements[1])
                antenna.frequency = devices["ANTENNAS"][elements[2]]["frequency"]
                antenna.height = devices["ANTENNAS"][elements[2]]["height"]
                antenna.group = elements[2]
                antenna.coords = (elements[3],elements[4])
                antenna.scenario = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["scenario"]
                antennas.append(antenna)
            if(elements[0] == "ue"):
                ue = UE(elements[1],elements[5])
                ue.height = devices["UES"][elements[2]]["height"]
                ue.group = elements[2]
                ue.coords = (elements[3],elements[4])
                ues.append(ue)
    return (antennas, ues)

##############################################
#           COORDINATES FUNCTIONS            #
##############################################

def gen_random_coords(terrain_shape: dict, n):
    shape = list(terrain_shape.keys())[0]
    length = terrain_shape[shape]['length']
    height = terrain_shape[shape]['height']
    coords = []
    for _ in range(n):
        coords.append([random.uniform(0, length), random.uniform(0, height)])
    return coords

def fill_up_the_lattice(N, lh, lv, nh, nv):
    def get_delta1d(L, n):
        return L / (n + 1)

    coords = []
    deltav = get_delta1d(lv, nv)
    deltah = get_delta1d(lh, nh)
    y = deltav
    count = 0
    while count < N:
        x = deltah
        current_nh = nh if count + nh <= N else N - count
        for _ in range(current_nh):
            coords.append((x, y))
            x += deltah
            count += 1
        y += deltav
    return coords

def get_rectangle_lattice_coords(lh, lv, N, Np, nh, nv):
    if Np > N:
        coords = fill_up_the_lattice(N, lh, lv, nh, nv)
    elif Np < N:
        coords = fill_up_the_lattice(N, lh, lv, nh, nv + 1)
    else:
        coords = fill_up_the_lattice(N, lh, lv, nh, nv)
    return coords

def gen_lattice_coords(terrain_shape: dict, N: int):
    shape = list(terrain_shape.keys())[0]
    lh = terrain_shape[shape]['length']
    lv = terrain_shape[shape]['height']
    R = lv / lh
    nv = round(math.sqrt(N / R))
    nh = round(R * nv)
    Np = nh * nv
    if shape.lower() == 'rectangle':
        coords = get_rectangle_lattice_coords(lh, lv, N, Np, nh, nv)
    else:
        msg = [f"\tImproper shape ({shape}) used in the\n",
               "\tgeneration of lattice coordinates.\n",
               "\tValid values: ['rectangle']"]
        ERROR(''.join(msg), 2)
    return coords

##############################################
#             PLOTTING FUNCTIONS             #
##############################################

def plot_transmissions(ue_data_frames, antenna_data_frames, tstart, tfinal, dt, ues, antennas, pdf_filename="ts_eq24_graphiques.pdf"):
    with PdfPages(pdf_filename) as pdf:
        ue_data_frames = np.array(ue_data_frames)
        antenna_data_frames = np.array(antenna_data_frames)

        numTicks = int((tfinal - tstart) / dt)
        time_slots = np.linspace(tstart, tfinal, numTicks + 1)

        # Histogramme des bits envoyés par tick (UEs)
        ue_avg_per_tick = np.mean(ue_data_frames, axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(time_slots[:-1], ue_avg_per_tick, width=dt, color='b', alpha=0.7, align='edge')
        plt.xlabel("Temps (s)")
        plt.ylabel("Moyenne des bits envoyés")
        plt.title("Moyenne des transmissions des UEs par tick")
        plt.xticks(time_slots, [f"{t:.1f}" for t in time_slots], rotation=45, fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        pdf.savefig()
        plt.close()

        # Histogramme des bits reçus par antenne
        max_subplots = 3  
        num_antennas = len(antennas)
        num_figures = (num_antennas + max_subplots - 1) // max_subplots

        for fig_idx in range(num_figures):
            fig, axes = plt.subplots(min(max_subplots, num_antennas - fig_idx * max_subplots), 1, 
                                     figsize=(12, 4 * max_subplots), sharex=True)

            if isinstance(axes, plt.Axes):
                axes = [axes]

            for i, antenna_idx in enumerate(range(fig_idx * max_subplots, min((fig_idx + 1) * max_subplots, num_antennas))):
                antenna = antennas[antenna_idx]
                axes[i].bar(time_slots[:-1], antenna_data_frames[antenna_idx], width=dt, color='r', alpha=0.7, align='edge')
                axes[i].set_ylabel("Bits reçus")
                axes[i].set_title(f"Réception de l'antenne {antenna.id}")
                axes[i].grid(axis="y", linestyle="--", alpha=0.7)
                axes[i].set_xticks(time_slots)
                axes[i].set_xticklabels([f"{t:.1f}" for t in time_slots], rotation=45, fontsize=9)

            axes[-1].set_xlabel("Temps (s)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Diagramme à barres du nombre total de bits reçus par chaque antenne
        total_bits_received = np.sum(antenna_data_frames, axis=1)  
        plt.figure(figsize=(12, 6))
        plt.bar([f"{antenna.id}" for antenna in antennas], total_bits_received, color='g', alpha=0.7)
        plt.xlabel("Antennes")
        plt.ylabel("Total des bits reçus")
        plt.title("Total des bits reçus par chaque antenne")
        plt.xticks(fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Boxplot des transmissions reçues par antenne
        plt.figure(figsize=(12, 6))
        plt.boxplot(antenna_data_frames, positions=np.arange(numTicks), widths=0.5)
        plt.xlabel("Tick")
        plt.ylabel("Bits reçus")
        plt.title("Distribution des bits reçus par tick (Antennes)")
        plt.xticks(np.arange(numTicks), [f"{t:.1f}" for t in time_slots[:-1]], rotation=45, fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        pdf.savefig()
        plt.close()

        # Heatmap des transmissions par antenne
        plt.figure(figsize=(12, 6))
        plt.imshow(antenna_data_frames, aspect='auto', cmap='Reds', interpolation='nearest')
        plt.colorbar(label="Bits reçus")
        plt.xlabel("Tick")
        plt.ylabel("Antennes")
        plt.title("Carte thermique des bits reçus par antenne")
        plt.xticks(np.arange(numTicks), [f"{t:.1f}" for t in time_slots[:-1]], rotation=45, fontsize=9)
        pdf.savefig()
        plt.close()

        # Scatter plot avec évolution des bits reçus par antenne
        plt.figure(figsize=(12, 6))
        for antenna_idx in range(len(antennas)):
            plt.scatter(time_slots[:-1], antenna_data_frames[antenna_idx], label=f"Antenne {antennas[antenna_idx].id}")
            plt.plot(time_slots[:-1], antenna_data_frames[antenna_idx], linestyle='-', alpha=0.7)
        plt.xlabel("Temps (s)")
        plt.ylabel("Bits reçus")
        plt.title("Évolution des transmissions par antenne")
        plt.legend()
        plt.grid(alpha=0.7)
        pdf.savefig()
        plt.close()

    print(f"Plots saved to {pdf_filename}")

##############################################
#           SIMULATION FUNCTIONS             #
##############################################

def compute_antenna_transmission(data_case, ue_data_frames, antennas, ues):
    tstart = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tstart"]
    tfinal = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tfinal"]
    dt = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["dt"]
    numTicks = int((tfinal - tstart)/dt)

    antenna_data_frames = [[0 for _ in range(numTicks)] for _ in range(len(antennas))]

    for ue in ues:
        for tick in range(1, numTicks + 1):
            antenna_data_frames[int(ue.assoc_ant)][tick - 1] += ue_data_frames[int(ue.id)][tick - 1]

    return antenna_data_frames

def generate_ue_transmission(data_case, devices, ues, antennas):
    tstart = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tstart"]
    tfinal = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tfinal"]
    dt = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["dt"]
    numTicks = int((tfinal - tstart) / dt)

    ue_data_frames = [[0 for _ in range(numTicks)] for _ in range(len(ues))]
    antenna_data_frames = [[0 for _ in range(numTicks)] for _ in range(len(antennas))]

    segment_file_name = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["read"]
    segments = read_segment_file(segment_file_name)

    for tick in range(1, numTicks + 1):
        for segment in segments:
            ue_id = segment[0]
            start = segment[1]
            end = segment[2]
            ue_R = devices["UES"][ues[ue_id].group]["R"]

            if start > dt * (tick - 1) and start < tick * dt:
                if end < tick * dt:
                    ue_data_frames[ue_id][tick - 1] += (end - start) * ue_R
                elif end > tick * dt:
                    ue_data_frames[ue_id][tick - 1] += ((tick * dt) - start) * ue_R
            elif start < dt * (tick - 1):
                if end > dt * (tick - 1) and end < tick * dt:
                    ue_data_frames[ue_id][tick - 1] += (end - (dt * (tick - 1))) * ue_R
                elif end > tick * dt:
                    ue_data_frames[ue_id][tick - 1] += ((tick * dt) - (dt * (tick - 1))) * ue_R

    with open("ts_eq24_transmission_ue.txt", "w") as file:
        for ue in ues:
            file.write(f"{ue.id}\n")
            for tick in range(numTicks):
                file.write(f"{float(tick)} {float(ue_data_frames[int(ue.id)][tick - 1])}\n")

    for ue in ues:
        for tick in range(1, numTicks + 1):
            antenna_data_frames[int(ue.assoc_ant)][tick - 1] += ue_data_frames[int(ue.id)][tick - 1]

    with open("ts_eq24_transmission_ant.txt", "w") as file:
        for antenna in antennas:
            file.write(f"{antenna.id}\n")
            for tick in range(numTicks):
                file.write(f"{float(tick)}: {float(antenna_data_frames[int(antenna.id)][tick - 1])}\n")

    return ue_data_frames, antenna_data_frames

def verify_equipment_validty(data_case,devices,ues,antennas):
    error_list = set()

    model = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["model"]
    scenario = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["scenario"]

    for antenna in antennas:
        antenna_frequency = devices["ANTENNAS"][antenna.group]["frequency"]
        antenna_height = devices["ANTENNAS"][antenna.group]["height"]
        if( model == "okumura"):
            if not(150 <= antenna_frequency*1000 <= 1500):
                error_list.add(f"Antenna of group {antenna.group} does not respect okumura frequency condition (150 to 1500 MHz)")
            if not(30 <= antenna_height <= 300):
                error_list.add(f"Antenna of group {antenna.group} does not respect okumura height condition (30 to 300 m)")
        elif ( model == "3gpp"):
            if( scenario == "RMa"):
                if not( 10 <= antenna_height <= 150):
                    error_list.add(f"Antenna of group {antenna.group} does not respect 3gpp-RMa height condition (10 to 150 m)")
                if not( 0.5 <= antenna_frequency <= 30):
                    error_list.add(f"Antenna of group {antenna.group} does not respect 3gpp-RMa frequency condition (0.5 to 30 GHz)")
            if( scenario == "UMa"):
                if not( 0.5 <= antenna_frequency <= 100):
                    error_list.add(f"Antenna of group {antenna.group} does not respect 3gpp-UMa frequency condition (0.5 to 100 GHz)")
            if(scenario == "UMi"):   
                if not( 0.5 <= antenna_frequency <= 100):
                    error_list.add(f"Antenna of group {antenna.group} does not respect 3gpp-UMi frequency condition (0.5 to 100 GHz)")
    for ue in ues:
        ue_height = devices["UES"][ue.group]["height"]
        if( model == "okumura"):
            if not(1 <= ue_height <= 10):
                error_list.add(f"UE of group {ue.group} does not respect okumura height condition (1 to 10 m)")
        elif ( model == "3gpp"):
            if (scenario == "RMa"):
                if not( 1 <= ue_height <= 10):
                    error_list.add(f"UE of group {ue.group} does not respect 3gpp-RMa height condition (1 to 10 m)")
            if (scenario == "UMa"):
                if not( 1.5 <= ue_height <= 22.5):
                    error_list.add(f"UE of group {ue.group} does not respect 3gpp-UMa height condition (1.5 to 22.5 m)")
            if (scenario == "UMi"):
                if not( 1.5 <= ue_height <= 22.5):
                    error_list.add(f"UE of group {ue.group} does not respect 3gpp-UMi height condition (1.5 to 22.5 m)")
    if (len(error_list) > 0) :
        print("\nList of errors regarding equipment validity depending on model:")
        for error in error_list:
            print(error)
        exit() 

def lab3(data_case,devices):

    antenna_names = []
    ue_names = []

    for device in data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"]:
        if device in devices.get("ANTENNAS", {}):
            antenna_names.append(device)
        elif device in devices.get("UES",{}):
            ue_names.append(device)
        else:
            msg = f"Device name {device} is not in device data base file"
            ERROR(msg, 1)

    antenna_amount = 0
    ue_amount = 0

    for antenna_type in range (len(antenna_names)):
        if(antenna_names[antenna_type] in data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"]):
            antenna_amount += data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"][antenna_names[antenna_type]]["number"]
    
    for ue_type in range (len(ue_names)):
        if(ue_names[ue_type] in data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"]):
            ue_amount += data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"][ue_names[ue_type]]["number"]
    
    if(data_case["ETUDE_DE_TRANSMISSION"]["ANT_COORD_GEN"] == "g"):
        antenna_coords = gen_lattice_coords(data_case["ETUDE_DE_TRANSMISSION"]["GEOMETRY"]["Surface"], antenna_amount)
    else:
        antenna_coords = gen_random_coords(data_case["ETUDE_DE_TRANSMISSION"]["GEOMETRY"]["Surface"], antenna_amount)

    if(data_case["ETUDE_DE_TRANSMISSION"]["UE_COORD_GEN"] == "g"):
        ue_coords = gen_lattice_coords(data_case["ETUDE_DE_TRANSMISSION"]["GEOMETRY"]["Surface"], ue_amount)
    else:
        ue_coords = gen_random_coords(data_case["ETUDE_DE_TRANSMISSION"]["GEOMETRY"]["Surface"], ue_amount)
    
    antennas = []
    ues = []

    coord_offset = 0
    for antenna_type in range(len(antenna_names)):
        if(antenna_names[antenna_type] in data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"]):
            for antennaId in range(data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"][antenna_names[antenna_type]]["number"]):
                antenna = Antenna(antennaId)
                antenna.frequency = devices["ANTENNAS"][antenna_names[antenna_type]]["frequency"]
                antenna.height = devices["ANTENNAS"][antenna_names[antenna_type]]["height"]
                antenna.group = antenna_names[antenna_type]
                antenna.coords = antenna_coords[antennaId + coord_offset]
                antenna.scenario = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["scenario"]
                antenna.gen = data_case["ETUDE_DE_TRANSMISSION"]["ANT_COORD_GEN"]
                antennas.append(antenna)
            coord_offset += data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"][antenna_names[antenna_type]]["number"]

    coord_offset = 0
    ueID_offset = 0
    for ue_type in range(len(ue_names)):
        if(ue_names[ue_type] in data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"]):
            for ueId in range(data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"][ue_names[ue_type]]["number"]):
                ue = UE(ueId+ueID_offset,f"app{ue_type+1}")
                ue.height = devices["UES"][ue_names[ue_type]]["height"]
                ue.group = ue_names[ue_type]
                ue.coords = ue_coords[ueId+coord_offset]
                ue.gen =  data_case["ETUDE_DE_TRANSMISSION"]["UE_COORD_GEN"]
                ues.append(ue)
            coord_offset += data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"][ue_names[ue_type]]["number"]
        ueID_offset = len(ues)    
    return (antennas,ues)

##############################################
#                    MAIN                    #
##############################################



def treat_cli_args(args) :
    if((len(args)) > 1 ):
        print("Insert only one argument")
        exit()
    if(os.path.isfile(os.path.join(os.getcwd(),args[0]))):
        pass
    else:
        print("No case file exists")
        exit()
    
    return args[0]

import matplotlib.pyplot as plt
from collections import defaultdict

def plottingFunction(antennas):
    SLOT_DURATION = 1e-3  # 1 ms per tick

    # Initialize per-tick per-app stats
    app_colors = {"app1": "steelblue", "app2": "orange", "app3": "forestgreen"}
    apps = ["app1", "app2", "app3"]
    app_packet_counts = defaultdict(lambda: {app: 0 for app in apps})
    app_bit_counts = defaultdict(lambda: {app: 0 for app in apps})

    for antenna in antennas:
        for tick, packets in enumerate(antenna.packet_queues_tick):
            for pkt in packets:
                app = pkt.app.lower()
                if app in apps:
                    app_packet_counts[tick][app] += 1
                    app_bit_counts[tick][app] += pkt.size

    ticks = sorted(app_packet_counts.keys())
    times = [tick * SLOT_DURATION * 1000 for tick in ticks]  # time in milliseconds

    # Plot packet count histogram per app
    plt.figure(figsize=(10, 5))
    bottom = [0] * len(ticks)
    for app in apps:
        values = [app_packet_counts[tick][app] for tick in ticks]
        plt.bar(times, values, bottom=bottom, width=1, label=app, color=app_colors[app])
        bottom = [bottom[i] + values[i] for i in range(len(values))]
    plt.xlabel("Tick")
    plt.ylabel("Number of Packets Transmitted")
    plt.title("Packet Transmission per Tick")
    plt.legend(loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plot total bits histogram per app
    plt.figure(figsize=(10, 5))
    bottom = [0] * len(ticks)
    for app in apps:
        values = [app_bit_counts[tick][app] for tick in ticks]
        plt.bar(times, values, bottom=bottom, width=1, label=app, color=app_colors[app])
        bottom = [bottom[i] + values[i] for i in range(len(values))]
    plt.xlabel("Tick")
    plt.ylabel("Total Bits Transmitted")
    plt.title("Bit Transmission per Tick")
    plt.legend(loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    for antenna in antennas:
        for tick, packets in enumerate(antenna.packet_queues_slot):
            for pkt in packets:
                app = pkt.app.lower()
                if app in apps:
                    app_packet_counts[tick][app] += 1
                    app_bit_counts[tick][app] += pkt.size

    ticks = sorted(app_packet_counts.keys())
    times = [tick * SLOT_DURATION * 1000 for tick in ticks]  # time in milliseconds

    # Plot packet count histogram per app
    plt.figure(figsize=(10, 5))
    bottom = [0] * len(ticks)
    for app in apps:
        values = [app_packet_counts[tick][app] for tick in ticks]
        plt.bar(times, values, bottom=bottom, width=1, label=app, color=app_colors[app])
        bottom = [bottom[i] + values[i] for i in range(len(values))]
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of Packets Transmitted")
    plt.title("Packet Transmission per Time Slot")
    plt.legend(loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plot total bits histogram per app
    plt.figure(figsize=(10, 5))
    bottom = [0] * len(ticks)
    for app in apps:
        values = [app_bit_counts[tick][app] for tick in ticks]
        plt.bar(times, values, bottom=bottom, width=1, label=app, color=app_colors[app])
        bottom = [bottom[i] + values[i] for i in range(len(values))]
    plt.xlabel("Time (ms)")
    plt.ylabel("Total Bits Transmitted")
    plt.title("Bit Transmission per Time Slot")
    plt.legend(loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main(args):
    random.seed(123)

    # Load YAML configuration files
    data_case = read_yaml_file("ts_eq24_cas.yaml")
    devices = read_yaml_file("devices_db.yaml")
    print("Case loaded")    

    #simulation time variables
    bw_mhz = data_case["ETUDE_DE_TRANSMISSION"]["FREQUENCY"]["BW"]
    scs_khz = data_case["ETUDE_DE_TRANSMISSION"]["FREQUENCY"]["SCS"] 
    dt = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["dt"]  # Durée d’un tick
    tstart = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tstart"]
    tfinal = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tfinal"]
    num_ticks = int((tfinal - tstart) / dt)
    slot_duration = 1.0 / (2 ** (math.log2(scs_khz / 15))) # en ms
    if slot_duration > dt or dt % slot_duration:
        ERROR("Il faut choisir un dt qui est a la fois plus grand et un multiple de la durée d'une slot")

    if("write" in data_case["ETUDE_DE_TRANSMISSION"]["COORD_FILES"]):
        # Create antenna and UE objects
        antennas, ues = lab3(data_case, devices)
        print("Devices created")
        create_text_file(data_case["ETUDE_DE_TRANSMISSION"]["COORD_FILES"]["write"], antennas,ues)
    else:
        #Read coord file
        (antennas,ues) = read_coord_file(data_case,devices)
        # Equipment validation
        verify_equipment_validty(data_case,devices,ues,antennas)
        verify_equipment_validty(data_case, devices, ues, antennas)
        print("Equipment verified")

    # Pathloss computation
    pathlosses = generate_pathlosses(data_case, ues, antennas)
    (minPl, maxPl) = findMinMaxPathloss("ts_eq24_pl.txt")
    print("Pathlosses calculated")

    # Association and pathloss recording
    create_assoc_files(data_case, ues, antennas, pathlosses)
    create_pathloss_file(data_case, ues, antennas, pathlosses)
    print("Recordings done")

    # Compute CQI and efficiency for each UE
    for ue in ues:
        pl = pathlosses[int(ue.id)][int(ue.assoc_ant)]
        ue.cqi = pathloss_to_cqi(pl, minPl, maxPl)
        ue.eff = get_efficiency_from_cqi(ue.cqi)
        print(f"\rUE efficiency: {ue.id}", end="")
    print("\n Efficiency calculated")

    # RB allocation
    antenna_weights = compute_antenna_load_weights(antennas, ues)
    total_nrb = get_nrb_from_bw_scs(bw_mhz, scs_khz)  # Assume 100 MHz and 30 kHz SCS
    assign_rb_proportionally(total_nrb, antenna_weights, antennas)
    print("RBs allocated")

    # Generation of traffic for UEs
    generate_packet_length_and_arrivals(data_case, devices, ues)
    print("\nUEs traffic generated")

    with open("pkt_size.txt", "w") as file:
        pass

    #Traffic Simulation
    current_time = 0
    slot_count = 0
    for tick in range(num_ticks+1):
        tick_start = tstart + tick * dt
        tick_end = tick_start + dt
        if tick != num_ticks+1:
            while current_time < tick_end:
                slot_traffic_creation(data_case, antennas, ues, current_time, tick)
                current_time += slot_duration  # assumed to be in ms
                slot_count += 1
            print(f"\rsimulation time: {current_time} ms", end="")
        else:
            print(f"\rsimulation time: {current_time} ms", end="")
        
    print("\nSimulation complete.")

    plottingFunction(antennas)

    # #petit test
    # for antenna in antennas:
    #     for packets_in_tick in antenna.packet_queues_tick:
    #         for packet in packets_in_tick:
    #             print(packet.size)    

if __name__ == '__main__':
    main(sys.argv[1:])