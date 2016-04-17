// Joshua Donnoe, Kyle Evens, and Dominik Haeflinger

#ifndef COMMON_H
#define COMMON_H

//
// Structs
//
typedef struct edge_t{
  unsigned short tree1;
  unsigned short tree2;
  float distance;
} edge_t;

typedef struct point_t {
  double coordinates[DIM];
} point_t;

//
//  Timing routines
//
double read_timer();

//
//  Argument processing
//
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );

#endif
