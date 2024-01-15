import numpy as np
import random
import math
from visualize import plot

MUTATION_RATE = 60
MUTATION_REPEAT_COUNT = 2
WEAKNESS_THRESHOLD = 850
nCities = 15

# Crea random le posizioni delle città in 2D
cityCoordinates = []
for i in range(nCities):
    cityCoordinates.append(np.random.randint(1,150,2))
    #cityCoordinates = [[5, 80], [124, 31], [46, 54], [86, 148], [21, 8],
    #               [134, 72], [49, 126], [36, 34], [26, 49], [141, 6],
    #               [124, 122], [80, 92], [70, 69], [76, 133], [23, 65]]

citySize = nCities

# Un genoma è un oggetto fatto dall'insieme dei cromosomi e dal suo fitness.
# In questo codice, infatti, vengono fatti evolvere i genomi. Ogni genoma è fatto
# da cromosomi, ogni cromosoma è una città (coppia di punti)
# Ogni genoma indica un percorso, che si calcola partendo dalla prima città fino
# ad arrivare all'ultima. L'ordine della sequenza delle città specifica un 
# percorso. Quindi ordini diversi (ovvero genomi diversi) indicano percorsi 
# diversi   
class Genome():
    chromosomes = []
    fitness = 9999

# Size indica il numero di genomi della popolazione
def CreateNewPopulation(size):
    population = []
    for x in range(size):
        newGenome = Genome()
        # Inizializza random la sequenza delle città per ogni cromosoma
        newGenome.chromosomes = random.sample(range(1, citySize), citySize - 1)
        # Aggiunge uno zero all'inizio
        newGenome.chromosomes.insert(0, 0)
        # Aggiunge uno zero alla fine
        newGenome.chromosomes.append(0)
        # Calcola il funzionale del cromosoma
        newGenome.fitness = Evaluate(newGenome.chromosomes)
        # Aggiunge il cromosoma alla popolaziones
        population.append(newGenome)
    return population


# Calculate distance between two point
def distance(a, b):
    dis = math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2))
    return np.round(dis, 2)

# Calcola il funzionale: la distanza tra una città e l'altra
def Evaluate(chromosomes):
    calculatedFitness = 0
    for i in range(len(chromosomes) - 1):
        # Va a prendere le coordinate delle città dalla mappa
        # cityCoordinates attraverso gli ID dei cromosomi
        p1 = cityCoordinates[chromosomes[i]]
        p2 = cityCoordinates[chromosomes[i + 1]]
        calculatedFitness += distance(p1, p2)
    calculatedFitness = np.round(calculatedFitness, 2)
    return calculatedFitness

# Restituisce il cromosoma con il funzionale più piccolo
def findBestGenome(population):
    allFitness = [i.fitness for i in population]
    bestFitness = min(allFitness)
    return population[allFitness.index(bestFitness)]


# Vengono selezionati casualmente k individui e preso il 
# migliore.
def TournamentSelection(population, k):
    # Seleziona random k individui
    selected = [population[random.randrange(0, len(population))] for i in range(k)]
    # Prende il migliore tra i k individui selezionati
    bestGenome = findBestGenome(selected)
    return bestGenome

# Restituisce un genoma figlio dopo aver effettuato un incrocio
# ottenuto tra due individui parent1 e parent1 e una successiva 
# mutazione
def Reproduction(population):
    parent1 = TournamentSelection(population, 10).chromosomes
    parent2 = TournamentSelection(population, 6).chromosomes
    while parent1 == parent2:
        parent2 = TournamentSelection(population, 6).chromosomes

    return OrderOneCrossover(parent1, parent2)

# Effettua un incrocio e una mutazione
# Esempio:
# parent1 = [0, 3, 8, 5, 1, 7, 12, 6, 4, 10, 11, 9, 2, 0]
# parent2 = [0, 1, 6, 3, 5, 4, 10, 2, 7, 12, 11, 8, 9, 0]
# child   = [0, 1, 3, 5, 2, 7, 12, 6, 4, 10, 11, 8, 9, 0]
def OrderOneCrossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size

    child[0], child[size - 1] = 0, 0

    # Seleziona randomicamente il punto di incrocio
    point = random.randrange(5, size - 4)

    for i in range(point, point + 4):
        child[i] = parent1[i]
    point += 4
    point2 = point
    while child[point] in [-1, 0]:
        if child[point] != 0:
            if parent2[point2] not in child:
                child[point] = parent2[point2]
                point += 1
                if point == size:
                    point = 0
            else:
                point2 += 1
                if point2 == size:
                    point2 = 0
        else:
            point += 1
            if point == size:
                point = 0
    
    # Effettua una mutazione del figlio ottenuto, con un tasso 
    # di mutazione dato da MUTATION_RATE 
    if random.randrange(0, 100) < MUTATION_RATE:
        child = SwapMutation(child)

    # Crea il genoma figlio
    newGenome = Genome()
    newGenome.chromosomes = child
    newGenome.fitness = Evaluate(child)
    return newGenome

# In questo caso la mutazione è fatta da uno scambio di città
# all'interno dello stesso genoma.
# Esempio:
# Genoma =         [0, 3, 8, 5, 1, 7, 12, 6, 4, 10, 11, 9, 2, 0]
# Genoma mutato =  [0, 11, 8, 5, 1, 7, 12, 6, 4, 10, 3, 9, 2, 0]
# Vengono effettuate MUTATION_REPEAT_COUNT mutazioni
def SwapMutation(chromo):
    for x in range(MUTATION_REPEAT_COUNT):
        p1, p2 = [random.randrange(1, len(chromo) - 1) for i in range(2)]
        while p1 == p2:
            p2 = random.randrange(1, len(chromo) - 1)
        log = chromo[p1]
        chromo[p1] = chromo[p2]
        chromo[p2] = log
    return chromo


def GeneticAlgorithm(popSize, maxGeneration):
    allBestFitness = []
    population = CreateNewPopulation(popSize)
    generation = 0
    # Ciclo principale
    while generation < maxGeneration:
        generation += 1

        # Incrocio e mutazione
        for i in range(int(popSize / 2)):
             population.append(Reproduction(population))

        # Selezione a soglia. Questo comporta un aumento della popolazione durante l'evoluzione
        for genom in population:
            if genom.fitness > WEAKNESS_THRESHOLD:
                population.remove(genom)

        # Fitness media della popolazione
        averageFitness = round(np.sum([genom.fitness for genom in population]) / len(population), 2)
        
        # Prende il miglior cromosoma
        bestGenome = findBestGenome(population)
        print("\n" * 1)
        print("Generation: {0}\nPopulation Size: {1}\t Average Fitness: {2}\nBest Fitness: {3}"
              .format(generation, len(population), averageFitness,
                      bestGenome.fitness))
        # Tiene traccia di tutte le migliori fitness di ogni generazione per poi 
        # tracciare la curva
        allBestFitness.append(bestGenome.fitness)
        
    # Visualize
    plot(generation, allBestFitness, bestGenome, cityCoordinates)
    


if __name__ == "__main__":
    GeneticAlgorithm(popSize=100, maxGeneration=300)
