## Écrire votre numéro d'équipe
## Érire les noms et matricules de chaque membre de l'équipe
## CECI EST OBLIGATOIRE
import sys
import math
import yaml
import random
#import simpy
import os
import pathloss_3gpp_eq24
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

class Antenna:
   def __init__(self, id):
       self.id = id           # id de l'antenne (int)
       self.frequency = None  # Fréquence de l'antenne en GHz
       self.height = None     # Hauteur de l'antenne
       self.group = None      # groupe défini dans la base de données (str)
       self.coords = None     # tuple contenant les coordonnées (x,y) 
       self.assoc_ues = []    # liste des ids des UEs associés à l'antenne
       self.scenario = None   # scénario de pathloss tel que lu du fichier de cas (str)
       self.gen = None        # type de génération de coordonnées: 'g', 'a', etc. (str)
       self.packet_queue = [] # tampon pour les paquets en attente de traitement
       self.all_packets = []  # liste de tous les paquets reçus
       self.current_slot = None  # Slot courante
       self.packets_this_slot = 0  # Paquets traités dans le slot courant
       self.bits_this_slot = 0    # Bits traités dans le slot courant
       self.scs = None        # Espacement des sous-porteuses (kHz)
       self.n_rb = None       # Nombre de blocs de ressources
       
   def __repr__(self):
       return f"Antenna(id={self.id}, freq={self.frequency}GHz, RBs={self.n_rb}, UEs={len(self.assoc_ues)})"
   
   def calculate_resource_blocks(self):
    """
    Détermine le nombre de RB (Ressource Blocks) disponibles en fonction
    de la largeur de bande du canal et l'espacement entre sous-porteuses.
    """
    # Déterminer si nous sommes en FR1 ou FR2
    is_fr2 = self.frequency > 6  # FR2 est au-dessus de 6 GHz
    
    # Largeur de bande par défaut selon la plage de fréquence
    # Dans un cas réel, cela devrait être lu à partir de la configuration
    if is_fr2:  # mmWave
        bandwidth_mhz = 100  # FR2 typiquement utilise 100 MHz
    else:
        bandwidth_mhz = 20   # FR1 typiquement utilise 20 MHz
    
    # Les valeurs ci-dessous sont extraites directement des normes 3GPP:
    # - TS 38.101-1 Table 5.3.2-1 pour FR1 (sub-6GHz)
    # - TS 38.101-2 Table 5.3.2-1 pour FR2 (mmWave)
    # Ces tableaux définissent le nombre exact de RBs pour chaque 
    # combinaison de largeur de bande et d'espacement de sous-porteuses.
    
    if is_fr2:  # Fréquences mmWave
        if self.scs == 60:
            if bandwidth_mhz == 50:
                self.n_rb = 66
            elif bandwidth_mhz == 100:
                self.n_rb = 132
            elif bandwidth_mhz == 200:
                self.n_rb = 264
            else:
                self.n_rb = 132  # Valeur par défaut
        elif self.scs == 120:
            if bandwidth_mhz == 50:
                self.n_rb = 32
            elif bandwidth_mhz == 100:
                self.n_rb = 66
            elif bandwidth_mhz == 200:
                self.n_rb = 132
            else:
                self.n_rb = 66  # Valeur par défaut
        else:
            self.n_rb = 132  # Défaut
            print(f"Avertissement: SCS {self.scs} non supporté pour FR2, utilisation de 132 RBs")
    else:  # Fréquences sub-6GHz
        if self.scs == 15:
            if bandwidth_mhz == 5:
                self.n_rb = 25
            elif bandwidth_mhz == 10:
                self.n_rb = 52
            elif bandwidth_mhz == 20:
                self.n_rb = 106
            elif bandwidth_mhz == 40:
                self.n_rb = 216
            else:
                self.n_rb = 106  # Défaut pour 20MHz
        elif self.scs == 30:
            if bandwidth_mhz == 5:
                self.n_rb = 11
            elif bandwidth_mhz == 10:
                self.n_rb = 24
            elif bandwidth_mhz == 20:
                self.n_rb = 51
            elif bandwidth_mhz == 40:
                self.n_rb = 106
            else:
                self.n_rb = 51  # Défaut pour 20MHz
        elif self.scs == 60:
            if bandwidth_mhz == 10:
                self.n_rb = 11
            elif bandwidth_mhz == 20:
                self.n_rb = 24
            elif bandwidth_mhz == 40:
                self.n_rb = 51
            else:
                self.n_rb = 24  # Défaut pour 20MHz
        else:
            self.n_rb = 106  # Défaut
            print(f"Avertissement: SCS {self.scs} non supporté pour FR1, utilisation de 106 RBs")
    
    print(f"Antenne {self.id}: {self.n_rb} blocs de ressources alloués (SCS: {self.scs} kHz, Bande: {bandwidth_mhz} MHz)")
           

   def receive_packet(self, env, packet):
       """
       Traite un paquet entrant
       
       Args:
           env: Environnement SimPy
           packet: Objet Packet à traiter
       """
       # Obtient l'UE qui a envoyé ce paquet
       ue = packet.source
       
       # Calcule les ressources disponibles
       overhead = 0  # Peut être 0, 6, 12, ou 18 (REs réservés)
       n_RE = 12 * 14 - overhead  # 12 sous-porteuses × 14 symboles - overhead, eqn 1 de l'énoncé
       n_RB = self.n_rb  # Nombre de blocs de ressources
       
       # Calcule le nombre maximum de bits d'information pouvant être transmis
       if ue.eff is None:
           # Défaut à une faible efficacité si non définie
           ue.eff = 0.1523  # Équivalent à CQI 1
           print(f"Avertissement: UE {ue.id} n'a pas d'efficacité définie, utilisation de la valeur par défaut")
       
       n_info = n_RB * n_RE * ue.eff  #eqn 2 de l'énoncé
       
       # Obtient le temps et le slot courants
       active_time = env.now
       dt = 1  # Durée d'un slot
       active_slot = int(active_time / dt)
       
       # Si nous avons changé de slot, traite les paquets en file d'attente
       if active_slot != self.current_slot:
           self.current_slot = active_slot
           bits_to_move = 0
           pacs_to_move = 0
           
           # Calcule combien de paquets nous pouvons traiter dans ce slot
           for pac in self.packet_queue:
               if (pac.size + bits_to_move) <= n_info:
                   bits_to_move += pac.size
                   pacs_to_move += 1
               else:
                   break
           
           self.packets_this_slot = pacs_to_move
           self.bits_this_slot = bits_to_move
           
           # Traite les paquets qui rentrent dans ce slot
           if pacs_to_move > 0:
               for pac in self.packet_queue[:pacs_to_move]:
                   pac.timeRX = active_slot * dt
                   self.all_packets.append(pac)
               
               # Retire les paquets traités de la file d'attente
               self.packet_queue = self.packet_queue[pacs_to_move:]
       
       # Traite le paquet courant
       if (self.bits_this_slot + packet.size) < n_info:
           # S'il rentre dans le slot courant, traite immédiatement
           self.bits_this_slot += packet.size
           self.packets_this_slot += 1
           packet.timeRX = env.now
           self.all_packets.append(packet)
       else:
           # Sinon, le met en file d'attente pour plus tard
           self.packet_queue.append(packet)


class Packet:
   def __init__(self, source, app, packet_id, packet_size, timeTX):
       self.id = packet_id
       self.size = packet_size
       self.timeTX = timeTX  # Temps d'envoi du paquet
       self.timeRX = None    # Temps de réception du paquet
       self.app = app
       self.source = source

class UE:
   def __init__(self, id, app_name):
       self.id = id           # id de l'UE (int)
       self.height = None     # Hauteur de l'UE
       self.group = None      # groupe défini dans la base de données (str)
       self.coords = None     # tuple contenant les coordonnées (x,y) 
       self.app = app_name    # nom de l'application exécutée sur l'UE (str)
       self.assoc_ant = None  # id de l'antenne associée à l'UE (int)
       self.los = True        # LoS ou non (bool)
       self.gen = None        # type de génération de coordonnées: 'g', 'a', etc. (str)
       self.packets = []      # liste des paquets générés par cet UE
       self.assoc_ant_pl = None  # Pathloss vers l'antenne associée
       self.cqi = None        # Indicateur de Qualité du Canal
       self.eff = None        # Efficacité spectrale
       self.packet_generator = None  # Processus SimPy pour la génération de paquets
   
   def __repr__(self):
       return f"UE(id={self.id}, app={self.app}, coords={self.coords}, ant={self.assoc_ant}, cqi={self.cqi})"
   
   def generate_packet(self, env, antennas):
       """
       Processus SimPy qui génère des paquets selon les caractéristiques de l'application
       
       Args:
           env: Environnement SimPy
           antennas: Liste de toutes les antennes dans la simulation
       """
       while True:
           packet_id = len(self.packets)  # Génère un ID de paquet unique
           
           if self.app == "Streaming4k":
               # Distribution exponentielle avec moyenne de 200ms
               yield env.timeout(random.expovariate(1.0 / (200e-3)))
               # Taille du paquet: 400 000 bits ±20%
               packet_size = int(random.uniform(0.8 * 400000, 1.2 * 400000))
               
           elif self.app == "Drone":
               # Distribution uniforme entre 30-40ms
               yield env.timeout(random.uniform(30e-3, 40e-3))
               # Taille du paquet: 100 bits ±5%
               packet_size = int(random.uniform(0.95 * 100, 1.05 * 100))
               
           elif self.app == "Auto_detect":
               # Distribution uniforme entre 700-1300ms
               yield env.timeout(random.uniform(700e-3, 1300e-3))
               # Taille du paquet: 100 bits ±5%
               packet_size = int(random.uniform(0.95 * 100, 1.05 * 100))
               
           else:
               raise ValueError(f"Type d'application inconnu: {self.app}")
           
           # Crée le paquet et l'ajoute à la liste des paquets de l'UE
           packet = Packet(self, self.app, packet_id, packet_size, env.now)
           self.packets.append(packet)
           
           # Envoie le paquet à l'antenne associée
           if self.assoc_ant is not None and 0 <= self.assoc_ant < len(antennas):
               antennas[self.assoc_ant].receive_packet(env, packet)
           else:
               print(f"Avertissement: UE {self.id} a une antenne associée invalide {self.assoc_ant}")


def fill_up_the_lattice(N, lh, lv, nh, nv):
    """Function appelée par get_rectangle_lattice_coords()"""

    def get_delta1d(L, n):
        return L/(n + 1)

    coords = []
    deltav = get_delta1d(lv, nv)
    deltah = get_delta1d(lh, nh)
    line = 1
    y = deltav
    count = 0
    while count < N:
        if count + nh < N:
            x = deltah
            for  i in range(nh):
                # Fill up the horizontal line
                coords.append((x,y))
                x = x + deltah
                count += 1
            line += 1
        else:
            deltah = get_delta1d(lh, N - count)
            x = deltah
            for i in range(N - count):
                # Fill up the last horizontal line
                coords.append((x,y))
                x = x + deltah
                count += 1
            line += 1
        y = y +deltav
    return coords

def get_rectangle_lattice_coords(lh, lv, N, Np, nh, nv):
    """Function appelee par gen_lattice_coords()"""

    if Np > N:
        coords = fill_up_the_lattice(N, lh, lv, nh, nv)
    elif Np < N:
        coords = fill_up_the_lattice(N, lh, lv, nh, nv + 1)
    else:
        coords = fill_up_the_lattice(N, lh, lv, nh, nv)
    return coords

def gen_lattice_coords(terrain_shape: dict, N: int):
    """Génère un ensemble de N coordonnées placées en grille 
       sur un terrain rectangulaire

       Args: terrain_shape: dictionary {'rectangle': {'length' : lh,
                                                   'height' : lv}
           lh and lv are given in the case file"""

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
                "\tgeneration of lattice coordinates.\n"
                "\tValid values: ['rectangle']"]
        ERROR(''.join(msg), 2)
    return coords        

def get_from_dict(key, data, res=None, curr_level = 1, min_level = 1):
    """Fonction qui retourne la valeur de n'importe quel clé du dictionnaire
       key: clé associé à la valeur recherchée
       data: dictionnaire dans lequel il faut chercher
       les autres sont des paramètres par défaut qu'il ne faut pas toucher"""
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

def okumura(scenario, frequency, distance, antenna_height, UE_height):
    if distance < 1:
        return 0
    if distance > 20:
        return math.inf
    # Constantes pour les corrections selon le scénario
    if scenario == "urban_small":
        correction_factor = 0
    elif scenario == "urban_large":
        correction_factor = -(-(1.1 * math.log10(frequency) - 0.7) * UE_height + (1.56 * math.log10(frequency) - 0.8))
        if(frequency <= 300):
            correction_factor -= 8.29*(math.log10(1.54*UE_height))**2 -1.1
        else:
            correction_factor -= 3.2*(math.log10(11.75*UE_height))**2 -4.97
    elif scenario == "suburban":
        correction_factor = -2 * (math.log10(frequency / 28))**2 - 5.4
    elif scenario == "open":
        correction_factor = -4.78 * (math.log10(frequency))**2 + 18.33 * math.log10(frequency) - 40.94
    else:
        raise ValueError("Scénario inconnu. Choisissez parmi: urban_large, urban_small, suburban, open.")

    # Modèle de base pour les grandes villes
    A = 69.55 + 26.16 * math.log10(frequency) - 13.82 * math.log10(antenna_height)
    B = (44.9 - 6.55 * math.log10(antenna_height)) * math.log10(distance)
    C = correction_factor

    pathloss = A + B + C - (1.1 * math.log10(frequency) - 0.7) * UE_height + (1.56 * math.log10(frequency) - 0.8)
    return pathloss

def define_los(filename, ue_id, antenna_id):
    ue_id = int(ue_id)
    antenna_id = int(antenna_id)
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                parts = line.split()
                first_column = int(parts[0])
                
                if first_column == ue_id:
                    # Check if the second number exists in the rest of the line
                    rest_of_line = map(int, parts[1:])  # Convert to integers
                    if antenna_id in rest_of_line:
                        return False
    return True

def threegpp(scenario, frequency, distance, antenna_height, UE_height, ue_id, antenna_id, visibility_file_name):
    los = define_los(visibility_file_name, ue_id, antenna_id)
    if(scenario == "RMa"):
        if(los):
            pathloss = pathloss_3gpp_eq24.rma_los(frequency,distance,antenna_height,UE_height)
        else:
            pathloss = pathloss_3gpp_eq24.rma_nlos(frequency,distance,antenna_height,UE_height)
    elif(scenario == "UMa"):
        if(los):
            pathloss = pathloss_3gpp_eq24.uma_los(frequency,distance,antenna_height,UE_height)
        else:
            pathloss = pathloss_3gpp_eq24.uma_nlos(frequency,distance,antenna_height,UE_height)
    elif(scenario == "UMi"):
        if(los):
            pathloss = pathloss_3gpp_eq24.umi_los(frequency,distance,antenna_height,UE_height)
        else:
            pathloss = pathloss_3gpp_eq24.umi_nlos(frequency,distance,antenna_height,UE_height)
    else:
        msg = f"Scenario name {scenario} is not available (RMa, UMa or UMi)"
        ERROR(msg, 1)
    return pathloss


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

# Fonction pour créer le fichier de pathloss ts_eq24_pl.txt
def create_pathloss_file(data_case, ues, antennas, pathlosses):
    model = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["model"]
    scenario = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["scenario"]
    with open("ts_eq24_pl.txt", "w") as file:
        for ue in ues:
            for antenna in antennas:
                file.write(f"{ue.id} {antenna.id} {pathlosses[int(ue.id)][int(antenna.id)]} {model} {scenario}\n")

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
            assoc_ues_str = " ".join(map(str, antenna.assoc_ues))
            file.write(f"{antenna.id} {assoc_ues_str}\n")

    # Créer le fichier d'association des UEs aux antennes
    with open("ts_eq24_assoc_ue.txt", "w") as file:
        for ue in ues:
            file.write(f"{ue.id} {ue.assoc_ant}\n")

def gen_random_coords(terrain_shape: dict, n):
    # Cette fonction doit générer les coordonées pour le cas de positionnement aléatoire
    # TODO
    shape = list(terrain_shape.keys())[0]
    length = terrain_shape[shape]['length']
    height = terrain_shape[shape]['height']
    coords = []
    for _ in range(n):
        coords.append([random.uniform(0,length),random.uniform(0,height)])
    return coords

def read_segment_file(fname):
    segments = []

    with open(fname,'r') as file:
        for line in file:
            elements = line.strip().split()
            segments.append([int(elements[0]),float(elements[1]),float(elements[2])])

    return segments

    
    #Do not use this function because R has been removed
def generate_ue_transmission(data_case,devices,ues,antennas):
    tstart = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tstart"]
    tfinal = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tfinal"]
    dt = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["dt"]
    numTicks = int((tfinal - tstart)/dt)

    ue_data_frames = [[0 for _ in range(numTicks)] for _ in range(len(ues))]
    antenna_data_frames = [[0 for _ in range(numTicks)] for _ in range(len(antennas))]

    segment_file_name = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["read"]

    segments = read_segment_file(segment_file_name)

    for tick in range(1,numTicks+1):
        for segment_line_index in range(len(segments)):
            ue_id = segments[segment_line_index][0]
            start = segments[segment_line_index][1]
            end = segments[segment_line_index][2]
            ue_R = devices["UES"][ues[ue_id].group]["R"]

            if  start > dt*(tick-1) and start < tick*dt:
                if end < tick*dt: #segment started and ended between tick n and n-1
                    ue_data_frames[ue_id][tick-1] += (end - start)*ue_R
                elif end > tick*dt: #segment started between tick n and n-1 but did not end in said tick
                    ue_data_frames[ue_id][tick-1] += ((tick*dt) - start)*ue_R
            elif start < dt*(tick-1):
                if end > dt*(tick-1) and end < tick*dt: #segment started before tick n-1 and stopped between tick n and n-1
                    ue_data_frames[ue_id][tick-1] += (end - (dt*(tick-1)))*ue_R
                elif end > tick*dt: #segement started before tick n-1 and will stop after tick n
                    ue_data_frames[ue_id][tick-1] += ((tick*dt) - (dt*(tick-1)))*ue_R

    with open("ts_eq24_transmission_ue.txt", "w") as file:
        for ue in ues:
            file.write(f"{ue.id}\n")
            for tick in range(numTicks):
                file.write(f"{float(tick)} {float(ue_data_frames[int(ue.id)][tick-1])}\n")

    #compute the antenna tranmssions
    for ue in ues:
        for tick in range(1,numTicks+1):
            antenna_data_frames[int(ue.assoc_ant)][tick-1] += ue_data_frames[int(ue.id)][tick-1]

    with open("ts_eq24_transmission_ant.txt", "w") as file:
        for antenna in antennas:
            file.write(f"{antenna.id}\n")
            for tick in range(numTicks):
                file.write(f"{float(tick)}: {float(antenna_data_frames[int(antenna.id)][tick-1])}\n")

    return ue_data_frames,antenna_data_frames

def compute_antenna_transmission(data_case,ue_data_frames,antennas,ues):
    tstart = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tstart"]
    tfinal = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tfinal"]
    dt = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["dt"]
    numTicks = int((tfinal - tstart)/dt)

    antenna_data_frames = [[0 for _ in range(numTicks)] for _ in range(len(antennas))]

    for ue in ues:
        for tick in range(1,numTicks+1):
            antenna_data_frames[int(ue.assoc_ant)][tick-1] += ue_data_frames[int(ue.id)][tick-1]

    return antenna_data_frames

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


def lab3 (data_case,devices):

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
    for ue_type in range(len(ue_names)):
        if(ue_names[ue_type] in data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"]):
            for ueId in range(data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"][ue_names[ue_type]]["number"]):
                ue = UE(ueId,f"app{ue_type+1}")
                ue.height = devices["UES"][ue_names[ue_type]]["height"]
                ue.group = ue_names[ue_type]
                ue.coords = ue_coords[ueId+coord_offset]
                ue.gen =  data_case["ETUDE_DE_TRANSMISSION"]["UE_COORD_GEN"]
                ues.append(ue)
            coord_offset += data_case["ETUDE_DE_TRANSMISSION"]["DEVICES"][ue_names[ue_type]]["number"]

    return (antennas,ues)

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

def ERROR(msg , code = 1):
    print("\n\n\nERROR\nPROGRAM STOPPED!!!\n")
    if msg:
        print(msg)
    print(f"\n\texit code = {code}\n\n\t\n")
    sys.exit(code)

def findMinMaxPathloss(plFileName):
    min_val=math.inf
    max_val=0
    with open(plFileName, "r") as f:
        for line in f:
            parts = line.strip().split()
            value = float(parts[2])
            min_val = min(min_val, value)
            max_val = max(max_val, value)
    return (min_val,max_val)

def pathloss_to_cqi(pathloss):
    
    (minPl, maxPl) = findMinMaxPathloss("ts_eq24_pl.txt")

    num_bins = 15
    step = (maxPl - minPl) / num_bins

    if pathloss < minPl:
        return num_bins
    if pathloss >= maxPl:
        return 0

    index = int((pathloss - minPl) / step)
    return num_bins - index 

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


def calculate_resource_blocks(bandwidth_mhz, subcarrier_spacing_khz):
    # Convert to Hz
    bandwidth_hz = bandwidth_mhz * 1e6
    subcarrier_spacing_hz = subcarrier_spacing_khz * 1e3
    
    # 12 subcarriers per RB
    rb_bandwidth = 12 * subcarrier_spacing_hz
    
    # Account for guard bands (90% usable)
    usable_bandwidth = bandwidth_hz * 0.9
    
    # Calculate number of RBs
    num_rbs = int(usable_bandwidth / rb_bandwidth)
    
    return num_rbs

def main(args):
    random.seed(123)

    data_case = read_yaml_file(treat_cli_args(args))

    model = data_case["ETUDE_DE_TRANSMISSION"]["PATHLOSS"]["model"]
    devices = read_yaml_file("devices_db.yaml")

    [antennas,ues] = lab3(data_case,devices)

    if("write" in data_case["ETUDE_DE_TRANSMISSION"]["COORD_FILES"]):
        create_text_file(data_case["ETUDE_DE_TRANSMISSION"]["COORD_FILES"]["write"], antennas,ues)
    else:
        tstart = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tstart"]
        tfinal = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["tfinal"]
        dt = data_case["ETUDE_DE_TRANSMISSION"]["CLOCK"]["dt"]

        

if __name__ == '__main__':
    main(sys.argv[1:])