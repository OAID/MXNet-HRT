#ifndef STD_IMPLEMENT_H
#define STD_IMPLEMENT_H
#include <string>
#include <sstream>
#if USE_ACL == 1
namespace std
{
    template < typename T > std::string to_string( const T& n )
    {
      std::ostringstream stm ;
      stm << n ;
      return stm.str() ;
    }
    inline float stof (const string& str, size_t* idx = 0)
    {
      float val;
      size_t len=sscanf(str.c_str(),"%f",&val);
      if (idx) *idx=len;
      return val; 
    }
    inline double stod (const string& str, size_t* idx = 0)
    {
      double val;
      size_t len=sscanf(str.c_str(),"%lf",&val);
      if (idx) *idx=len;
      return val; 
    }
    inline int stoi  (const string& str, size_t* idx = 0, int base = 10)
    {
      int val;
      size_t len=sscanf(str.c_str(),"%i",&val);
      if (idx) *idx=len;
      return val; 
    }
    inline long stol  (const string& str, size_t* idx = 0, int base = 10)
    {
      int val;
      size_t len=sscanf(str.c_str(),"%i",&val);
      if (idx) *idx=len;
      return val; 
    }
}
#endif
#endif
