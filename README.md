Hadoop-ReliefF
==============

Hadoop/MapReduce implementation of ReliefF feature selection method. Uses kd-trees for nearest neighbor search.

Input format: csv, where rows are records/observations, columns are attributes/features and the last column is the class

To run in Hadoop: 
1. Compile kdTree.java and ReliefF.java into a .jar.
2. Run the .jar in Hadoop. 
  Format: ReliefF input_location_in_HDFS output_location_in_HDFS number_of_nearest_neighbors sample_size number_of_maps
  
  Example: ReliefF input output 5 500000 32
