/** 
    @file GAEvalData.h
	@brief the object that contains evaluation data.
    
	@author Matthew Wall  
	@date 3-Dec-1995
	
  Copyright (c) 1995 Massachusetts Institute of Technology, all rights reserved.


  This is the basic interface for the object that contains evaluation data.  It
can be used with genomes and/or populations in combination with their
respective evaluation methods.
*/
 
#ifndef _ga_eval_h_
#define _ga_eval_h_

class GAEvalData
{
public:
    GAEvalData() {}
    GAEvalData(const GAEvalData&) {}
    virtual ~GAEvalData() {}
    GAEvalData& operator=(const GAEvalData& orig)
    {
        if(&orig != this)
        {
            copy(orig);
        }
        return *this;
    }
    virtual GAEvalData* clone() const = 0;
    virtual void copy(const GAEvalData&) = 0;
};

#endif
