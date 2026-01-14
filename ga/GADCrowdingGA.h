/** 
  @file GADCrowdingGA.h
  @brief   Header file for the steady-state genetic algorithm class.

  @author Matthew Wall  
  @date 29-Mar-1999

  Copyright (c) 1999 Matthew Wall  , all rights reserved
*/
 
#ifndef _ga_deterministic_crowding_ga_h_
#define _ga_deterministic_crowding_ga_h_

#include <ga/GABaseGA.h>

class GADCrowdingGA : public GAGeneticAlgorithm
{
public:
GADCrowdingGA(const GAGenome& g) : GAGeneticAlgorithm(g) {}
    virtual ~GADCrowdingGA() {}

    virtual void initialize(unsigned int seed = 0);
    virtual void step();
    GADCrowdingGA& operator++()
    {
        step();
        return *this;
    }
};

#endif
