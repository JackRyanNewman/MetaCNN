Jack Newman. 
10/1/2025

This folder contains my compiler experiments, conducted outside of my main C project. My focus was on exploring how bfloat works, and its interaction with SIMMD Libaries, and an simple implemntation. I also studied and dug into how the C compiler interacts with vectorization and threading under different conditions. In particular, I was interested in how the compiler handles prefetching on large datasets, and how different access patterns. 

I should note, I HEAVILY profiled, researched, and spent alot of time optimizing the data structures, and static implemtnations of the the main CNN file. 