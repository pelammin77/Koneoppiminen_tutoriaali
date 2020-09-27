"""
file: GA_test.py
Author: Petri Lamminaho
Desc: Simple Genetic algorithm. Genertates string 
"""
import random

POP_SIZE = 100
CHARS = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

TARGET_STRING = "Hi my name is Pete"


class Genetic_String:

    def __init__(self, str):
        self.str = str
        self.err = self.calculate_err()

    @classmethod
    def create_starting_string(cls):
        str_len = len(TARGET_STRING)
        return [cls.mutate()for _ in range(str_len)]



    def crossover(self, parent2):
        offspring = []
        for gs1, gs2 in zip(self.str, parent2.str):
            probability = random.random()

            if probability < 0.45:
                offspring.append(gs1)

            elif probability < 0.90:
                offspring.append(gs2)

            else:
                offspring.append(self.mutate())

        return Genetic_String(offspring)

    @classmethod
    def mutate(cls):
        return random.choice(CHARS)




    def calculate_err(self):

        err = 0
        for gs, ts in zip(self.str, TARGET_STRING):
            if gs != ts:
                err +=1

        return err


def main():
    generation = 1

    done = False
    population = []

    for _ in range(POP_SIZE):
        string = Genetic_String.create_starting_string()
        population.append(Genetic_String(string))

    while not done:
        population = sorted(population, key=lambda x: x.err)
        if population[0].err<=0:
            done = True
            break

        new_generation = []
        s = int((10 * POP_SIZE) / 100)
        new_generation.extend(population[:s])

        s = int((90 * POP_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.crossover(parent2)
            new_generation.append(child)

        population = new_generation

        print("Generation: {}\tString: {}\tError: {}". \
                format(generation,
                        "".join(population[0].str),
                        population[0].err))

        generation += 1

    print("Generation: {}\tString: {}\tError: {}". \
            format(generation,
                    "".join(population[0].str),
                    population[0].err))



if __name__ == '__main__':
    main()




