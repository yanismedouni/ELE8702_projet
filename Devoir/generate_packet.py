import random
import yaml

def generate_packets(app_config, num_packets=10):
    """
    Génère une série de paquets en utilisant la configuration d'une application. 

    Exemple de 10 paquets en streaming 4k.
    
    Cette fonction lit les paramètres liés aux inter-arrivées et à la taille des paquets 
    définis dans le dictionnaire de configuration (qui provient du fichier YAML de l'application).
    
    Args:
      app_config (dict): Dictionnaire contenant la configuration de l'application.
                         Exemple de format :
                           {
                               "interarrival": {
                                   "distribution": "exponential",
                                   "mean_ms": 200
                               },
                               "packet_length": {
                                   "distribution": "uniform",
                                   "base_bits": 400000,
                                   "variability": 0.20
                               }
                           }
      num_packets (int): Nombre de paquets à générer (par défaut 10).
      
    Returns:
      tuple: (interarrival_times, packet_lengths)
             interarrival_times est une liste des temps d'inter-arrivées en millisecondes,
             packet_lengths est une liste des longueurs de paquets en bits.
    """
    # Récupère les configurations spécifiques aux inter-arrivées et à la longueur des paquets
    interarrival_config = app_config.get("interarrival", {})
    packet_config = app_config.get("packet_length", {})
    
    interarrival_times = []
    packet_lengths = []
    
    # --- Génération des temps d'inter-arrivées ---
    arrival_distribution = interarrival_config.get("distribution", "uniform")
    
    if arrival_distribution == "exponential":
        # Pour une distribution exponentielle, la moyenne doit être définie
        mean_ms = interarrival_config.get("mean_ms")
        if not mean_ms:
            raise ValueError("La moyenne (mean_ms) doit être spécifiée pour une distribution exponentielle.")
        # Génère num_packets temps d'inter-arrivée suivant une loi exponentielle
        for _ in range(num_packets):
            # La fonction expovariate attend le paramètre lambda = 1/mean
            interarrival_times.append(random.expovariate(1.0 / mean_ms))
    elif arrival_distribution == "uniform":
        # Pour une distribution uniforme, il faut définir les bornes min_ms et max_ms
        min_ms = interarrival_config.get("min_ms")
        max_ms = interarrival_config.get("max_ms")
        if min_ms is None or max_ms is None:
            raise ValueError("min_ms et max_ms doivent être spécifiés pour une distribution uniforme.")
        for _ in range(num_packets):
            interarrival_times.append(random.uniform(min_ms, max_ms))
    else:
        raise ValueError("Type de distribution inconnu pour les inter-arrivées: " + arrival_distribution)
    
    # --- Génération des longueurs des paquets ---
    # Ici, nous utilisons uniquement une distribution uniforme autour de la valeur de base
    length_distribution = packet_config.get("distribution", "uniform")
    if length_distribution == "uniform":
        base_bits = packet_config.get("base_bits")
        variability = packet_config.get("variability", 0)
        if base_bits is None:
            raise ValueError("base_bits doit être spécifié pour la génération des longueurs de paquets.")
        # Calcul des bornes de la distribution en appliquant la variabilité
        lower_bound = base_bits * (1 - variability)
        upper_bound = base_bits * (1 + variability)
        for _ in range(num_packets):
            packet_lengths.append(random.uniform(lower_bound, upper_bound))
    else:
        # Possibilité d'ajouter d'autres types de distribution si nécessaire
        raise ValueError("Type de distribution inconnu pour la longueur des paquets: " + length_distribution)
    
    # Retourne deux listes : l'une pour les inter-arrivées et l'autre pour la longueur des paquets
    return interarrival_times, packet_lengths

def main():
    # Chargement du fichier YAML contenant la configuration des applications
    with open("apps.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Exemple : génération des paquets pour l'application "streaming_4k"
    streaming_config = config["applications"]["streaming_4k"]
    interarrivals, pkt_lengths = generate_packets(streaming_config, num_packets=10)
    
    # Affiche les résultats pour vérification
    print("Temps d'inter-arrivées (ms) :", interarrivals)
    print("Longueur des paquets (bits) :", pkt_lengths)

if __name__ == "__main__":
    main()
