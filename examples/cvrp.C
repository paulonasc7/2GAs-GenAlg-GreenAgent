#include <ga/ga.h>
#include <ga/GA1DArrayGenome.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// Simple capacitated VRP instance with a single depot and fixed vehicle capacity.
struct Customer {
  double x;
  double y;
  int demand;
};

struct CVRPInstance {
  Customer depot;
  std::vector<Customer> customers;
  int vehicleCapacity;
  int maxVehicles;
  double overVehiclePenalty;
  double overCapacityPenalty;
};

static double euclidean(const Customer &a, const Customer &b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

// Build a permutation for the genome: 0..N-1 shuffled.
void permutationInit(GAGenome &g) {
  GA1DArrayGenome<int> &genome = (GA1DArrayGenome<int> &)g;
  const int n = genome.length();
  for (int i = 0; i < n; ++i)
    genome.gene(i, i);
  for (int i = n - 1; i > 0; --i) {
    const int j = GARandomInt(0, i);
    const int tmp = genome.gene(i);
    genome.gene(i, genome.gene(j));
    genome.gene(j, tmp);
  }
}

// Decode a permutation into routes that respect capacity; return total cost with penalties.
float evaluateGenome(GAGenome &g) {
  GA1DArrayGenome<int> &genome = (GA1DArrayGenome<int> &)g;
  const CVRPInstance &inst = *(CVRPInstance *)genome.userData();

  double totalDistance = 0.0;
  int load = 0;
  int vehiclesUsed = 1;
  Customer last = inst.depot;

  for (int i = 0; i < genome.length(); ++i) {
    const Customer &cust = inst.customers[genome.gene(i)];
    // Start a new route if adding this customer would exceed capacity.
    if (load + cust.demand > inst.vehicleCapacity) {
      totalDistance += euclidean(last, inst.depot); // close current route
      vehiclesUsed++;
      load = 0;
      last = inst.depot;
    }
    totalDistance += euclidean(last, cust);
    load += cust.demand;
    last = cust;
  }

  totalDistance += euclidean(last, inst.depot); // return from final route

  // Penalize solutions that need too many vehicles or overfill capacity.
  if (vehiclesUsed > inst.maxVehicles) {
    totalDistance += inst.overVehiclePenalty * (vehiclesUsed - inst.maxVehicles);
  }

  // Penalize any residual overload on the final route.
  if (load > inst.vehicleCapacity) {
    totalDistance += inst.overCapacityPenalty * (load - inst.vehicleCapacity);
  }

  return static_cast<float>(totalDistance);
}

// Convert genome into a human-readable set of routes given capacities.
std::string renderRoutes(const GA1DArrayGenome<int> &genome, const CVRPInstance &inst) {
  std::ostringstream oss;
  int load = 0;
  int routeIdx = 1;
  oss << "Routes (capacity " << inst.vehicleCapacity << "):\n";
  oss << "Route " << routeIdx << ": depot -> ";
  for (int i = 0; i < genome.length(); ++i) {
    const Customer &cust = inst.customers[genome.gene(i)];
    if (load + cust.demand > inst.vehicleCapacity) {
      oss << "depot\n";
      routeIdx++;
      load = 0;
      oss << "Route " << routeIdx << ": depot -> ";
    }
    load += cust.demand;
    oss << "C" << genome.gene(i) << "(d" << cust.demand << ") -> ";
  }
  oss << "depot\n";
  return oss.str();
}

int main(int argc, char **argv) {
  // Parse custom operator flags before handing off to the GA parameter parser.
  std::string crossoverOpt = "order";
  std::string mutatorOpt = "swap";
  std::string selectorOpt = "roulette";
  std::vector<char *> forwarded;
  forwarded.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--crossover" && i + 1 < argc) {
      crossoverOpt = argv[++i];
      continue;
    }
    if (arg == "--mutator" && i + 1 < argc) {
      mutatorOpt = argv[++i];
      continue;
    }
    if (arg == "--selector" && i + 1 < argc) {
      selectorOpt = argv[++i];
      continue;
    }
    forwarded.push_back(argv[i]);
  }
  forwarded.push_back(nullptr);
  int argc2 = static_cast<int>(forwarded.size()) - 1;
  char **argv2 = forwarded.data();

  GAAlleleSet<int> alleleSet;
  const CVRPInstance instance{
      /*depot*/ {0.0, 0.0, 0},
      /*customers*/
      {{2, 4, 2},   {8, 1, 3},   {5, 6, 4},   {6, 3, 2},   {1, 7, 5},
       {3, 8, 2},   {7, 8, 3},   {9, 5, 4},   {4, 2, 2},   {8, 6, 3},
       {2, 9, 4},   {6, 9, 2},   {10, 2, 5},  {11, 6, 3},  {12, 4, 4},
       {3, 5, 1},   {5, 1, 2},   {9, 9, 5},   {11, 9, 3},  {12, 8, 2}},
      /*vehicleCapacity*/ 15,
      /*maxVehicles*/ 5,
      /*overVehiclePenalty*/ 50.0,
      /*overCapacityPenalty*/ 25.0};

  const int numCustomers = instance.customers.size();
  for (int i = 0; i < numCustomers; ++i)
    alleleSet.add(i);

  GA1DArrayAlleleGenome<int> genome(numCustomers, alleleSet, evaluateGenome, (void *)&instance);
  genome.initializer(permutationInit);

  // Choose crossover based on CLI option.
  if (crossoverOpt == "onepoint") {
    genome.crossover(GA1DArrayGenome<int>::OnePointCrossover);
  } else if (crossoverOpt == "twopoint") {
    genome.crossover(GA1DArrayGenome<int>::TwoPointCrossover);
  } else if (crossoverOpt == "partialmatch") {
    genome.crossover(GA1DArrayGenome<int>::PartialMatchCrossover);
  } else if (crossoverOpt == "cycle") {
    genome.crossover(GA1DArrayGenome<int>::CycleCrossover);
  } else if (crossoverOpt == "uniform") {
    genome.crossover(GA1DArrayGenome<int>::UniformCrossover);
  } else if (crossoverOpt == "EvenOdd") {
    genome.crossover(GA1DArrayGenome<int>::EvenOddCrossover);
  } else { // default: order crossover for permutations
    genome.crossover(GA1DArrayGenome<int>::OrderCrossover);
  }

  // Choose mutator based on CLI option (only swap provided by default).
  genome.mutator(GA1DArrayGenome<int>::SwapMutator);
  (void)mutatorOpt; // placeholder if more mutators are added later.

  GASimpleGA ga(genome);
  // ga.minimize();
  // ga.maximize();

  // Register defaults and let GA parse standard CLI flags.
  GAParameterList params;
  GASimpleGA::registerDefaultParameters(params);
  params.set(gaNpopulationSize, 100);
  params.set(gaNnGenerations, 200);
  params.set(gaNpMutation, 0.03);
  params.set(gaNpCrossover, 0.9);
  params.set(gaNnBestGenomes, 1);
  params.set(gaNnConvergence, 20);
  params.set(gaNscoreFrequency, 1);
  params.set(gaNflushFrequency, 20);
  params.set(gaNelitism, gaTrue);
  params.parse(argc2, argv2);
  ga.parameters(params);

  ga.minimize();
  // Persist resolved GA parameters for auditing.
  params.write("cvrp_params.txt");

  // Selector choice (custom flag).
  GARouletteWheelSelector rwsel;
  GATournamentSelector toursel;
  GARankSelector ranksel;
  if (selectorOpt == "tournament") {
    ga.selector(toursel);
  } else if (selectorOpt == "rank") {
    ga.selector(ranksel);
  } else {
    ga.selector(rwsel);
  }

  ga.recordDiversity(gaTrue);

  std::ofstream statfile("cvrp_stats.tsv");
  statfile << "generation\tmean\tmax\tmin\tdeviation\tdiversity\n";

  ga.initialize();
  while(!ga.done())
  {
    const GAStatistics &s = ga.statistics();
    statfile << s.generation() << "\t"
             << s.current(GAStatistics::Mean) << "\t"
             << s.current(GAStatistics::Maximum) << "\t"
             << s.current(GAStatistics::Minimum) << "\t"
             << s.current(GAStatistics::Deviation) << "\t"
             << s.current(GAStatistics::Diversity) << "\n";
    ga.step();
  }
  // Log the terminal generation after the final step.
  const GAStatistics &s = ga.statistics();
  statfile << s.generation() << "\t"
           << s.current(GAStatistics::Mean) << "\t"
           << s.current(GAStatistics::Maximum) << "\t"
           << s.current(GAStatistics::Minimum) << "\t"
           << s.current(GAStatistics::Deviation) << "\t"
           << s.current(GAStatistics::Diversity) << "\n";
  statfile.close();

  GAStatistics &stats = (GAStatistics &)ga.statistics(); // const_cast for file output

  // Summarize the run and best solution.
  const GA1DArrayGenome<int> &best = (const GA1DArrayGenome<int> &)stats.bestIndividual();
  const double bestScore = best.score();

  std::ofstream summary("cvrp_summary.txt");
  summary << "best_cost\t" << bestScore << "\n";
  summary << "maxEver\t" << stats.maxEver() << "\n";
  summary << "minEver\t" << stats.minEver() << "\n";
  summary << "offlineMax\t" << stats.offlineMax() << "\n";
  summary << "offlineMin\t" << stats.offlineMin() << "\n";
  summary << "online\t" << stats.online() << "\n";
  summary << "convergence\t" << stats.convergence() << "\n";
  summary << "generations\t" << stats.generation() << "\n";
  summary << "selections\t" << stats.selections() << "\n";
  summary << "crossovers\t" << stats.crossovers() << "\n";
  summary << "mutations\t" << stats.mutations() << "\n";
  summary << "replacements\t" << stats.replacements() << "\n";
  summary.close();

  std::ofstream routeOut("cvrp_best_routes.txt");
  routeOut << "Best score (lower is better): " << bestScore << "\n";
  routeOut << renderRoutes(best, instance);
  routeOut.close();

  std::cout << "Finished CVRP run, stats written to cvrp_stats.tsv\n";
  return 0;
}
