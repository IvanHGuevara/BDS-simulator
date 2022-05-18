import operator
import math
import random
import numpy
from datetime import datetime
from datetime import timedelta
from typing import List

from ems.algorithms.selection.ambulance_selection import AmbulanceSelector
from ems.datasets.travel_times.travel_times import TravelTimes
from ems.models.ambulances.ambulance import Ambulance
from ems.models.cases.case import Case

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# An implementation of a "fastest travel time" ambulance_selection from a base to
# the demand point closest to a case
class EvolvableSelector(AmbulanceSelector):

    def __init__(self,
                 travel_times: TravelTimes = None):
        self.travel_times = travel_times
        # Genetic Programming constants:
        self.POPULATION_SIZE = 2000
        self.P_CROSSOVER = 0.9
        self.P_MUTATION = 0.01
        self.MAX_GENERATIONS = 2
        self.HALL_OF_FAME_SIZE = 10
        self.MIN_TREE_HEIGHT = 3
        self.MAX_TREE_HEIGHT = 5
        self.LIMIT_TREE_HEIGHT = 17
        self.MUT_MIN_TREE_HEIGHT = 0
        self.MUT_MAX_TREE_HEIGHT = 2
        self.NUM_INPUTS = 6
        self.NUM_COMBINATIONS = 2 ** self.NUM_INPUTS
        self.pset = self.createGPModel()

    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1
    
    def runGP(self, time):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        toolbox.register("evaluate", self.evalSymbReg, time)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox = toolbox
        population = toolbox.population(n=self.POPULATION_SIZE)
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=self.P_CROSSOVER, mutpb=self.P_MUTATION,
                                               ngen=self.MAX_GENERATIONS, stats=mstats,
                               halloffame=hof, verbose=False)     
        return logbook.chapters['fitness'].select("min")

    def evalSymbReg(self, ambulanceTime, individual):
        # Transform the tree expression in a callable function
        print("Ambulance total time: ", ambulanceTime.total_seconds())
        print("Individual :", individual)
        func = self.toolbox.compile(expr=individual)
        try:
            value = func(ambulanceTime.total_seconds())
        except TypeError:
            value = 0
        print("The result value is:", value)
        return value,

    def createGPModel(self):
        pset = gp.PrimitiveSet("MAIN", 1) # number of inputs!!!
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(self.protectedDiv, 2) 
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.sin, 1)
        pset.addEphemeralConstant("rand101", lambda: random.random())
        return pset

    def select_ambulance(self,
                         available_ambulances: List[Ambulance],
                         case: Case,
                         current_time: datetime):
        
        ambulances_times = []
        loc_set_2 = self.travel_times.destinations
        closest_loc_to_case, _, _ = loc_set_2.closest(case.incident_location)

        for amb in available_ambulances:

            # Compute closest location in the first set to the ambulance
            ambulance_location = amb.location

            # Compute the time from the location point mapped to the ambulance to the location point mapped to the case
            time = self.travel_times.get_time(ambulance_location, closest_loc_to_case)

            ambulances_times.append(
                (time, amb)
            )
        # Dummy logic to deliver a result
        for amb_time in ambulances_times:
            minValue = self.runGP(amb_time[0])
               
        ambulanceSelected = ambulances_times[0][1]
        return ambulanceSelected