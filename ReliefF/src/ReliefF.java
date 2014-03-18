import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.util.ArrayList;
import java.util.Random;

public final class ReliefF {

	private static int runReliefF(String[] args) throws Exception{
		Configuration conf = new Configuration();
		conf.set("input",args[0]);
		conf.set("k",args[2]);
		conf.set("iterNum",args[3]);
		//conf.set("mapred.child.java.opts", "-Xss32m");

		Job job = new Job(conf);
		job.setJobName("Relief feature selection");

		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(FloatWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(FloatWritable.class);

		job.setMapperClass(ReliefFMap.class);
		job.setReducerClass(ReliefFReduce.class);
		job.setCombinerClass(ReliefFReduce.class);

		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(args[1]), true);
		
		//TextInputFormat.setMinInputSplitSize(job, 300*1024);
		if(args.length==5){
			int numMaps = Integer.parseInt(args[4]);
			Path path = new Path("data");
			FileStatus fstatus[] = fs.listStatus(path);
			long datasetSize = 0;
			for(FileStatus f: fstatus){
				if(f.getPath().toUri().getPath().contains("/_"))
					continue;
				datasetSize += f.getLen();
			}
			long mapSplitSize = (long) Math.floor((double) datasetSize / numMaps);
			TextInputFormat.setMaxInputSplitSize(job, mapSplitSize);
		}

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setJarByClass(ReliefF.class);


		TextInputFormat.setInputPaths(job,"data");
		TextOutputFormat.setOutputPath(job,new Path(args[1]));
		job.setNumReduceTasks(1);

		System.out.println("Starting ReliefF job");
		int exitcode = job.waitForCompletion(true) ? 0 : 1;
		if(exitcode != 0){
			System.out.println("ReliefF failed");
			return exitcode;
		}
		else{

			ArrayList<Integer> keys = new ArrayList<Integer>();
			ArrayList<Float> weights = new ArrayList<Float>();

			Path path = new Path(args[1]);
			FileStatus fstatus[] = fs.listStatus(path);

			for(FileStatus f: fstatus){
				if(f.getPath().toUri().getPath().contains("/_"))
					continue;

				FSDataInputStream input = fs.open(f.getPath());
				BufferedReader br1 = new BufferedReader(new InputStreamReader(input));

				String currLine = br1.readLine();
				while(currLine != null){
					String[] line1 = currLine.split("\\s+");
					keys.add(Integer.parseInt(line1[0]));
					weights.add(Float.parseFloat(line1[1]));
					currLine = br1.readLine();
				}

				br1.close();						
			}

			int m = keys.size();
			float[][] result = new float[m][2];
			for(int i=0; i<m; i++){
				result[i][0] = keys.get(i);
				result[i][1] = weights.get(i);
			}

			//sort result
			for(int i=0; i <m-1; i++){
				int maxIndex = i;
				for(int j=i+1; j<m; j++){
					if(result[j][1] > result[maxIndex][1])
						maxIndex = j;
				}
				float temp = result[i][1];
				int tempIndex = (int) result[i][0];
				result[i][1] = result[maxIndex][1];
				result[i][0] = result[maxIndex][0];
				result[maxIndex][1] = temp;
				result[maxIndex][0] = tempIndex;
			}

			System.out.println("\n\nattribute, weight");
			for(int i=0; i < m; i++)
				System.out.println((int) result[i][0] + ", " + result[i][1]/*/(Integer.parseInt(args[3])*Integer.parseInt(args[2]))*/);

			return exitcode;
		}
	}

	private static int generateSubset(String[] args) throws Exception{

		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(args[1]), true);

		//LOAD INPUT TO GENERATE SUBSET
		Path inpath = new Path(args[0]);
		FileStatus fstatus[] = fs.listStatus(inpath);

		ArrayList<String> dataset = new ArrayList<String>(1000);

		for(FileStatus f: fstatus){
			if(f.getPath().toUri().getPath().contains("/_"))
				continue;

			FSDataInputStream input = fs.open(f.getPath());
			BufferedReader br = new BufferedReader(new InputStreamReader(input));

			String currLine = br.readLine();
			while(currLine != null){
				dataset.add(currLine);
				currLine = br.readLine();
			}
			br.close();
		}

		System.out.print("Classes: ");
		ArrayList<String> classesList = new ArrayList<String>(2);
		for(String currLine : dataset){
			String[] temp = currLine.replaceAll(" ", "").split(",");
			String tempclass = temp[temp.length-1];
			if(tempclass != ""){
				boolean newclass = true;
				for(String currclass : classesList){
					if(currclass.equals(tempclass)){
						newclass = false;
						break;
					}				
				}
				if(newclass){
					classesList.add(tempclass);
					System.out.print(tempclass + " ");
				}
			}
		}	

		System.out.println("\nNumber of classes: " + classesList.size());

		//GENERATE SUBSET
		System.out.println("Generating subset");
		int totalSize = dataset.size();
		int subsetSize = Integer.parseInt(args[3]);
		if(subsetSize >= totalSize)
			subsetSize = totalSize;
		
		String[] subset = new String[subsetSize];
		Random generator = new Random();
		
		for(int i=0; i < subsetSize; i++)
			subset[i] = dataset.remove(generator.nextInt(dataset.size()));

		System.out.println("Storing subset");
		//STORE SUBSET
		FileSystem outfs = FileSystem.get(URI.create(args[0]+"/subset"), conf);
		FSDataOutputStream out = outfs.create(new Path("data"));
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out));

		for(int i=0; i < subset.length; i++){
			bw.write(subset[i]);
			bw.newLine();
		}
		bw.flush();
		bw.close();

		return classesList.size();
	}

	private static int runKdTree(String[] args, int numClasses) throws Exception{
		Configuration conf = new Configuration();
		conf.set("input",args[0]);

		Job job = new Job(conf);
		job.setJobName("kdTree builder");

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		job.setMapperClass(KdTreeMap.class);
		job.setReducerClass(KdTreeReduce.class);

		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(args[1]), true);
		
		if(args.length==5){
			int numMaps = Integer.parseInt(args[4]);
			Path path = new Path(args[0]);
			FileStatus fstatus[] = fs.listStatus(path);
			long datasetSize = 0;
			for(FileStatus f: fstatus){
				if(f.getPath().toUri().getPath().contains("/_"))
					continue;
				datasetSize += f.getLen();
			}
			long mapSplitSize = (long) Math.floor((double) datasetSize / numMaps);
			TextInputFormat.setMaxInputSplitSize(job, mapSplitSize);
		}

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setJarByClass(ReliefF.class);

		TextInputFormat.setInputPaths(job,args[0]);
		TextOutputFormat.setOutputPath(job,new Path(args[1]));

		job.setNumReduceTasks(numClasses);

		System.out.println("Generating kdTrees");
		int exitcode = job.waitForCompletion(true) ? 0 : 1;
		if(exitcode != 0){
			System.out.println("kdTree generation failed!");
			return exitcode;
		}
		else{
			return exitcode;
		}
	}

	private static ArrayList<kdTree> LoadKdTrees(Configuration conf) throws IOException{
		FileSystem fs = FileSystem.get(conf);
		Path temp = new Path("trees");
		FileStatus fstatus[] = fs.listStatus(temp);

		ArrayList<kdTree> kdTreeList = new ArrayList<kdTree>();

		for(FileStatus f: fstatus){
			if(!f.getPath().toUri().getPath().contains(".ser"))
				continue;

			FSDataInputStream input = fs.open(f.getPath());
			ObjectInputStream in = new ObjectInputStream(input);

			try {
				kdTreeList.add((kdTree) in.readObject());				
			} 
			catch (ClassNotFoundException e) {
				System.out.println("Error reading class tree");
			}

			in.close();
		}
		return kdTreeList;
	}

	private static void storeKdTree(kdTree classTree, Configuration conf, String className) throws IOException{
		FileSystem fs = FileSystem.get(conf);
		Path temp = new Path("trees/"+className+".ser");
		ObjectOutputStream out = new ObjectOutputStream(fs.create(temp));
		out.writeObject(classTree);
		out.close();
	}

	public static int main(String[] args) throws Exception {

		if(args.length < 4 || args.length > 5){
			System.out.println("ReliefF <input dataset location> <output location> <number of neighbors> <num of iterations> <size of map in KB>");
			System.exit(-1);
		}

		long StartTime = System.currentTimeMillis();

		//GENERATE SUBSET
		int numClasses = generateSubset(args);

		//GENERATE KDTREES
		int exitcode = runKdTree(args,numClasses);

		//RUN RELIEFF
		if(exitcode == 0){
			exitcode = runReliefF(args);
		}

		float elapsedTime = (float) (System.currentTimeMillis() - StartTime)/1000;
		System.out.println("Elapsed time: " + ((int)(elapsedTime/3600))%60 + "hr " + ((int)(elapsedTime/60))%60 + "min " + elapsedTime%60 + "s");

		return exitcode;
	}

	public static class ReliefFMap extends Mapper<LongWritable, Text, IntWritable, FloatWritable> {

		//k is the number of nearest neighbors to locate
		int k;
		//m is the number of attributes including class
		int m;
		//nominal indicates which attribute is nominal
		boolean nominal[];
		//iterNum is the samplesize
		int iterNum;
		//classProbabilityList stores the probability of each class occurring
		ArrayList<Float> classProbabilityList;
		//treeClassList stores the classes of each tree
		ArrayList<String> treeClassList;
		//treeList stores the kdtree of each class
		ArrayList<kdTree> treeList;

		protected void setup(Context context) throws IOException, InterruptedException{

			Configuration conf = context.getConfiguration();

			k = Integer.parseInt(conf.get("k"));
			iterNum = Integer.parseInt(conf.get("iterNum"));

			treeList = LoadKdTrees(conf);

			m = treeList.get(0).getM();
			nominal = treeList.get(0).getNominal();
			
			treeClassList = new ArrayList<String>(2);
			int total = 0;
			for(kdTree currTree : treeList){
				treeClassList.add(currTree.getClassName());
				total += currTree.getNumLeaves();
			}

			classProbabilityList = new ArrayList<Float>(2);
			for(kdTree currTree : treeList)
				classProbabilityList.add(((float) currTree.getNumLeaves())/total);

		}

		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			String[] stringRecord = value.toString().replaceAll(" ", "").split(",");

			//skip record if it has missing values
			boolean skip = false;
			if(stringRecord.length == m){	
				for(int i=0; i<m; i++){
					if(stringRecord[i].length()==0 || stringRecord[i].contains("?")){
						skip = true;
						break;
					}
				}
			}
			
			if(stringRecord.length == m && !skip){

				String obsClass = stringRecord[m-1];

				int currClassTreeIndex = -1;
				for(int j=0; j < treeClassList.size(); j++)
					if(treeClassList.get(j).equals(obsClass)){
						currClassTreeIndex = j;
						break;
					}

				float[] min = treeList.get(currClassTreeIndex).getmin();
				float[] range = treeList.get(currClassTreeIndex).getRange();

				float[] observation = new float[m-1];
				float[] attrWeight = new float[m-1];

				//normalize
				for(int i=0; i<m-1; i++){	

					if(nominal[i])
						observation[i] = (stringRecord[i].hashCode()-min[i])/range[i];
					else
						observation[i] = (Float.parseFloat(stringRecord[i])-min[i])/range[i];
				}



				//find k near hits
				ArrayList<float[]> nearHits = treeList.get(currClassTreeIndex).knn(observation, k);				
				float[] currNearHit;
				for(int j=0; j < nearHits.size(); j++){
					currNearHit = nearHits.get(j);
					for(int z=0; z < m-1; z++){
						if(nominal[z] && observation[z] != currNearHit[z]){
							attrWeight[z] -= 1;
						}
						else{
							float diff = Math.abs(observation[z] - currNearHit[z]);
							attrWeight[z] -= diff;
						}

					}
				}

				//find k near misses from each class
				for(int c=0; c < treeList.size(); c++)
					if(c != currClassTreeIndex){
						ArrayList<float[]> nearMisses = treeList.get(c).knn(observation, k);
						float[] currNearMiss;
						for(int j=0; j < nearMisses.size(); j++){
							currNearMiss = nearMisses.get(j);
							for(int z=0; z < m-1; z++){
								if(nominal[z] && observation[z] != currNearMiss[z]){
									attrWeight[z] += (classProbabilityList.get(c)/(1-classProbabilityList.get(currClassTreeIndex)));
								}
								else{
									float diff = Math.abs(observation[z] - currNearMiss[z]);
									attrWeight[z] += (classProbabilityList.get(c)/(1-classProbabilityList.get(currClassTreeIndex)))*diff;
								}	
							}
						}
					}

				//emit <key,value> pairs
				for(int i=0; i<m-1; i++){
					context.write(new IntWritable(i+1),new FloatWritable(attrWeight[i]/(iterNum*k)));
				}
			}
		}
	}

	public static class ReliefFReduce extends Reducer<IntWritable, FloatWritable, IntWritable, FloatWritable> {

		public void reduce(IntWritable key, Iterable<FloatWritable> values, Context context) throws IOException, InterruptedException {

			float finalWeight = 0;
			for(FloatWritable recordWeight : values)
				finalWeight += recordWeight.get();

			context.write(key, new FloatWritable(finalWeight));
		}
	}

	public static class KdTreeMap extends Mapper<LongWritable, Text, Text, Text> {

		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] stringRecord = value.toString().replaceAll(" ", "").split(",");
			context.write(new Text(stringRecord[stringRecord.length-1]),value);
		}

	}

	public static class KdTreeReduce extends Reducer<Text, Text, Text, Text> {

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

			Configuration conf = context.getConfiguration();

			//dataset stores the float representation of the input dataset
			ArrayList<float[]> dataset = new ArrayList<float[]>(1000);
			//stringDataset stores the string[] representation of the input dataset
			ArrayList<String[]> stringDataset = new ArrayList<String[]>(1000);
			//m is the number of attributes + class;
			int m;
			//used to calculate range
			float[] min;
			float[] max;
			//range stores the range of each attribute
			float[] range;
			//nominal indicates which attributes are nominal
			boolean[] nominal;

			for(Text currLine : values)
				stringDataset.add(currLine.toString().replaceAll(" ","").split(","));

			//INITIALIZE VALUES OF SOME VARIABLES
			m = stringDataset.get(0).length;
			while(m == 0){
				stringDataset.remove(0);
				m = stringDataset.get(0).length;
			}

			min = new float[m-1];
			max = new float[m-1];
			for(int i=0; i < m-1; i++){
				min[i] = Float.POSITIVE_INFINITY;
				max[i] = Float.NEGATIVE_INFINITY;
			}

			nominal = new boolean[m-1];
			String[] tempRecord = stringDataset.get(0);
			for(int i = 0; i < m-1; i++){
				try{
					//throws exception if record[i] is not numeric
					Float.parseFloat(tempRecord[i]);
					nominal[i] = false;
				}

				//attribute i is nominal
				catch(NumberFormatException e){
					nominal[i] = true;
				}
			}


			for(String[] stringRecord : stringDataset){

				if(stringRecord.length == m){

					//skip if record is incomplete
					boolean skip = false;
					for(int i=0; i<m; i++){
						if(stringRecord[i].length()==0 || stringRecord[i].contains("?")){
							skip = true;
							break;
						}
					}
					if(skip)					
						continue;


					float[] floatRecord = new float[m-1];
					//convert String to float
					for(int i=0; i<m-1; i++){
						if(nominal[i])
							floatRecord[i] = stringRecord[i].hashCode();

						else
							floatRecord[i] = Float.parseFloat(stringRecord[i]);

						//update max
						if(floatRecord[i] > max[i])
							max[i] = floatRecord[i];

						//update min
						if(floatRecord[i] < min[i])
							min[i] = floatRecord[i];
					}
					dataset.add(floatRecord);
				}
			}

			range = new float[m-1];
			for(int i=0; i < m-1; i++){
				range[i] = max[i] - min[i];
				if(range[i]==0)
					range[i]=1;
			}

			//normalize dataset to [0,1]
			for(int i=0; i < dataset.size(); i++){
				float[] floatRecord = dataset.get(i);
				for(int j=0; j < m-1; j++){
					floatRecord[j] = (floatRecord[j]-min[j])/range[j];
				}
			}

			//build kd-tree
			kdTree classtree = new kdTree(dataset, key.toString(), nominal, range, min);

			//store kd-tree
			storeKdTree(classtree, conf, key.toString());
		}
	}
}