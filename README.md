# MapReduce-framework

MapReduce is a programming model and an associated implementation for processing and generating large data sets with a parallel, distributed algorithm on a cluster. A MapReduce program is composed of a Map() procedure that performs filtering and sorting, and a Reduce() procedure that performs a summary operation. The MapReduce framework orchestrates the processing by marshalling the distributed servers, running the various tasks in parallel, managing all communications and data transfers between the various parts of the system, and providing for redundancy and fault tolerance.

## Files

```
MapReduceFramework.cpp
MapReduceFramework.h
MapReduceClient.h
Barrier.cpp
Barrier.h
SampleClient.cpp
```

## How to use

In order to use this framework it is necessary to implement the API of MapReduceClient.h.
An example is provided in the SampleClient.cpp

## More Info

This project was done as part of Operating Systems coarse in the Hebrew university of Jerusalem.
It is a multiprocessing program that can be used for different jobs. As an example counting the number of timed each letter appear in some text.
