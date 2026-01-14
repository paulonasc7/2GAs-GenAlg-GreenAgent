/** 
  @file GDId.h
  @brief This defines the identifiers for polymorphic classes.
  
  @author Matthew Wall 
  @date 7-May-1995

  Copyright (c) 1995 Massachusetts Institute of Technology, all rights reserved.

  The IDs are used primarily for checking to be see if the two objects are the same type before
doing a copy, for example.  The name is often used when printing out error
messages so you'll know where things are happening.

  I hate to derive so many classes from the same base class, especially when
the derived classes are completely unrelated.  But this is a convenient way to
enumerate the built-in classes, and they DO share the polymorphic behaviour
(even if they do NOT share any other attributes).


 @author Yan Ren <pemryan@gmail.com>
 @date   2012-10-30  
 copyright (c) 2012 Enintech, Inc.
 
 replaced GAID's classID/className implementation with RTTI.
*/

#ifndef _ga_id_h_
#define _ga_id_h_

#include <typeinfo>

class GAID
{
public:

    int sameClass(const GAID &b) const
    {
        return(typeid(*this) == typeid(b));
    }
    
	const char * className() const
    {
        return typeid(*this).name();
    }
    	
    virtual ~GAID() { }
};

#endif

