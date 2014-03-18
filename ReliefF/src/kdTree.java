import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

public class kdTree implements Serializable {

	private static final long serialVersionUID = 3446733123816327493L;
	
	private node root; 
	private int internalSize;
	private int numLeaves;
	private ArrayList<node> nodeList;

	private KNNList knnList; 
	private float[] observation;

	private boolean[] nominal;
	private float[] range;
	private float[] min;
	private int m;
	private String className;
	
	public float[] getRange(){
		return this.range;
	}
	
	public float[] getmin(){
		return this.min;
	}
	
	public int getM(){
		return this.m;
	}
	
	public String getClassName(){
		return this.className;
	}

	public boolean[] getNominal(){
		return this.nominal;
	}
	
	public kdTree(ArrayList<float[]> dataset, String className, boolean[] nominal, float[] range, float[] min) throws IllegalArgumentException{
		if(dataset == null || dataset.size() < 1)
			throw new IllegalArgumentException("Please provide a dataset.");
		else{
			this.nodeList = new ArrayList<node>();
			this.root = new node(this.nodeList);
			int totalAttr = dataset.get(0).length;
			buildTree(this.root.getNodeId(), dataset, 0, totalAttr);
			this.nominal = nominal;
			this.range = range;
			this.min = min;
			this.m = range.length + 1;
			this.className = className;
		}
	}

	private void buildTree(int rootId, ArrayList<float[]> dataset, int attrIndex, int totalAttr){

		node root = this.nodeList.get(rootId);

		if(dataset.size() != 1){
			float median = findMedian(dataset, attrIndex, (int) Math.floor(dataset.size()/2));
			ArrayList<ArrayList<float[]>> separatedData = this.separateData(dataset, median, attrIndex);

			int searchCount = 0;
			while(separatedData.get(0).size() == 0 && searchCount < totalAttr){
				attrIndex = (attrIndex + 1)%totalAttr;
				median = findMedian(dataset, attrIndex, (int) Math.floor(dataset.size()/2));
				separatedData = this.separateData(dataset, median, attrIndex);
				searchCount++;
			}

			if(separatedData.get(0).size() == 0){
				root.setValues(dataset.get(0));
				numLeaves++;
			}
			else{
				this.internalSize+=2;

				root.setMedian(median);
				root.setAttrIndex(attrIndex);			

				root.addChildren(this.nodeList);
				this.buildTree(root.getLeft().nodeId, separatedData.get(0), (attrIndex + 1)%totalAttr, totalAttr);
				this.buildTree(root.getRight().nodeId, separatedData.get(1), (attrIndex + 1)%totalAttr, totalAttr);	
			}
		}
		else{
			root.setValues(dataset.get(0));		
			numLeaves++;
		}
	}

	private float findMedian(ArrayList<float[]> data, int attrIndex, int k){
		if (data.size() <= 10){
			float[] tempData = new float[data.size()];
			for(int i = 0; i < data.size(); i++)
				tempData[i] = data.get(i)[attrIndex];
			Arrays.sort(tempData);
			if(k == data.size())
				return tempData[k-1];
			else
				return tempData[k];
		}

		else{
			int n = data.size();

			//separate data in n/5 groups of 5 
			int numGroups = (int) Math.ceil(n/5);
			ArrayList<ArrayList<float[]>> groupList = new ArrayList<ArrayList<float[]>>(numGroups);
			int processed = 0;		
			for(int i = 0; i < numGroups; i++){
				ArrayList<float[]> arr = new ArrayList<float[]>(5);
				for(int j = 0; j < 5; j++){
					if(processed == n)
						break;
					else
						arr.add(data.get(processed++));

				}
				groupList.add(arr);
			}

			//find median in all groups
			ArrayList<float[]> groupMedians = new ArrayList<float[]>(numGroups);
			float[] tempArray = new float[1];
			for(int i = 0; i < numGroups; i++){
				tempArray[0] = this.findMedian(groupList.get(i),attrIndex,3);
				groupMedians.add(tempArray);
			}

			//find median of medians
			float median = this.findMedian(groupMedians, 0, n/10);

			//partition data into left<median, equal=median, right>median
			ArrayList<ArrayList<float[]>> partionedData = this.partition(data, median, attrIndex);
			//median is in left partition
			if(k <= partionedData.get(0).size())
				return findMedian(partionedData.get(0), attrIndex, k);
			//median is in right partition
			else if(k > partionedData.get(0).size() + partionedData.get(1).size())
				return findMedian(partionedData.get(2), attrIndex, k - partionedData.get(0).size() - partionedData.get(1).size());
			//median found
			else
				return median;
		}
	}	

	private ArrayList<ArrayList<float[]>> partition(ArrayList<float[]> dataset, float median, int attrIndex){		
		ArrayList<ArrayList<float[]>> partitionedData = new ArrayList<ArrayList<float[]>>(3);
		partitionedData.add(new ArrayList<float[]>());
		partitionedData.add(new ArrayList<float[]>());
		partitionedData.add(new ArrayList<float[]>());

		for(int i=0; i < dataset.size(); i++){
			if(dataset.get(i)[attrIndex] < median)
				partitionedData.get(0).add(dataset.get(i));
			else if(dataset.get(i)[attrIndex] == median)
				partitionedData.get(1).add(dataset.get(i));
			else
				partitionedData.get(2).add(dataset.get(i));
		}

		return partitionedData;
	}

	private ArrayList<ArrayList<float[]>> separateData(ArrayList<float[]> dataset, float median, int attrIndex){

		ArrayList<ArrayList<float[]>> separatedData = new ArrayList<ArrayList<float[]>>(2);
		separatedData.add(new ArrayList<float[]>());
		separatedData.add(new ArrayList<float[]>());

		for(int i=0; i < dataset.size(); i++){
			if(dataset.get(i)[attrIndex] < median)
				separatedData.get(0).add(dataset.get(i));
			else
				separatedData.get(1).add(dataset.get(i));
		}

		return separatedData;
	}

	public node getRoot(){
		return this.root;
	}

	public int getInternalSize(){
		return this.internalSize;
	}

	public int getNumLeaves(){
		return this.numLeaves;
	}

	private node nearestNeighbor(float[] observation){
		int attrIndex;
		float compareValue;
		node currNode = this.root;		
		while(!currNode.isLeaf()){
			attrIndex = currNode.getAttrIndex();
			compareValue = currNode.getMedian();
			if(observation[attrIndex] < compareValue)
				currNode = currNode.getLeft();
			else
				currNode = currNode.getRight();
		}
		return currNode;
	}

	public ArrayList<float[]> knn(float[] observation, int k){

		this.knnList = new KNNList(k);
		this.observation = observation;

		auxknn(this.nearestNeighbor(observation).getNodeId());
		ArrayList<float[]> result = this.knnList.getNeighbors();
		this.knnList = null;
		this.observation = null;
		this.unvisitNodes();
		return result;

	}

	private void unvisitNodes() {
		for(int i =0; i < this.nodeList.size(); i++)
			this.nodeList.get(i).unvisit();
	}

	private void auxknn(int subtreeRootIndex){

		node subtreeRoot = this.nodeList.get(subtreeRootIndex);

		//reached a possible nearest neighbor
		if(subtreeRoot.isLeaf()){
			float currDistanceToObs = 0;
			float[] currValues = subtreeRoot.getValues();
			for(int i =0 ; i < observation.length; i++){
				float tempDist = observation[i] - currValues[i];
				currDistanceToObs += tempDist*tempDist;
			}
			if(currDistanceToObs != 0){
				this.knnList.add(currValues, currDistanceToObs);
			}
			subtreeRoot.visit();
			subtreeRoot = subtreeRoot.getParent();
			subtreeRootIndex = subtreeRoot.getNodeId();
		}

		//check if subtree contains possible nearest neighbors
		boolean checkLeft = true;
		boolean checkRight = true;
		if(knnList.isFull()){
			float subTreeMinDistToObs;
			if(nominal[subtreeRoot.getAttrIndex()]){
				subTreeMinDistToObs = 1;
				if(subTreeMinDistToObs >= knnList.getKthDistanceToObs()){
					checkLeft = false;
					checkRight = false;
				}
			}
			//left tree contains possible nearest neighbors
			else if(observation[subtreeRoot.getAttrIndex()] > subtreeRoot.getMedian()){
				subTreeMinDistToObs = Math.abs(observation[subtreeRoot.getAttrIndex()] - subtreeRoot.getMedian());
				if(subTreeMinDistToObs >= knnList.getKthDistanceToObs())
					checkRight = false;
			}	
			//right tree contains possible nearest neighbors
			else if(observation[subtreeRoot.getAttrIndex()] <= subtreeRoot.getMedian()){
				subTreeMinDistToObs = Math.abs(subtreeRoot.getMedian() - observation[subtreeRoot.getAttrIndex()]);
				if(subTreeMinDistToObs >= knnList.getKthDistanceToObs())
					checkLeft = false;
			}	
		}


		//check left
		node nextSubtree = subtreeRoot.getLeft();
		if(checkLeft && nextSubtree != null && !nextSubtree.isVisited()){
			auxknn(nextSubtree.getNodeId());
			nextSubtree.visit();
		}

		//check right
		nextSubtree = subtreeRoot.getRight();
		if(checkRight && nextSubtree != null && !nextSubtree.isVisited()){
			auxknn(nextSubtree.getNodeId());		
			nextSubtree.visit();	
		}

		//move up the tree
		subtreeRoot.visit();
		if(subtreeRoot.getParent() != null)
			auxknn(subtreeRoot.getParent().getNodeId());

	}

	public String toString(){
		String result = "";
		return toString(this.root, result);
	}

	private String toString(node currNode, String result){
		if(currNode.isLeaf())
			result += currNode.toString();
		else
			result += currNode.toString() + "\n" + this.toString(currNode.getLeft(),result) + "\n" + toString(currNode.getRight(), result);
		return result;
	}

	public class node implements Serializable{

		private static final long serialVersionUID = 7345940230906697621L;
		private float[] values;
		private node parent;
		private node left;
		private node right;

		private float median;
		private int attrIndex;

		private boolean visited;

		private int nodeId;

		public node(float[] values, ArrayList<node> nodeList){
			this.nodeId = nodeList.size();
			this.values = values;	
			this.visited = false;
			nodeList.add(this);
		}

		public int getNodeId() {
			return this.nodeId;
		}

		public node(float median, int attrIndex, ArrayList<node> nodeList){
			this.nodeId = nodeList.size();
			this.median = median;
			this.attrIndex = attrIndex;
			this.visited = false;
			nodeList.add(this);
		}

		public node(node parent, ArrayList<node> nodeList){
			this.nodeId = nodeList.size();
			this.parent = parent;
			this.visited = false;
			nodeList.add(this);
		}

		public node(ArrayList<node> nodeList){
			this.nodeId = nodeList.size();
			this.visited = false;
			nodeList.add(this);
		};

		public float distanceTo(node target){
			if(target == null || target.getValues() == null){
				return Float.POSITIVE_INFINITY;
			}
			else{
				float[] targetValues = target.getValues();
				float distance = 0;
				for(int i = 0; i < targetValues.length; i++)
					distance += Math.abs(this.values[i]-targetValues[i]);
				return distance;
			}
		}		

		public node getParent(){
			return this.parent;
		}

		public void setParent(node parent){
			this.parent = parent;
		}

		public void addChildren(ArrayList<node> nodeList){
			this.left = new node(this, nodeList);
			this.right = new node(this, nodeList);
		}

		public float[] getValues(){
			return this.values;
		}

		public void setValues(float[] values){
			this.values = values;
		}

		public node getLeft(){
			return this.left;
		}

		public node getRight(){
			return this.right;
		}

		public void setLeft(node left){
			this.left = left;
		}

		public void setRight(node right){
			this.right = right;
		}

		public boolean isLeaf(){
			return this.left==null && this.right==null;
		}

		public float getMedian(){
			return this.median;
		}

		public int getAttrIndex(){
			return this.attrIndex;
		}

		public void setMedian(float median){
			this.median = median;
		}

		public void setAttrIndex(int attrIndex){
			this.attrIndex = attrIndex;
		}

		public void visit(){
			this.visited = true;
		}

		public void unvisit(){
			this.visited = false;
		}

		public boolean isVisited(){
			return this.visited;
		}

		public String toString(){
			String result;
			if(this.isLeaf()){
				result = "({";
				for(int i=0; i < this.values.length-1; i++){
					result += this.values[i] + ", ";
				}
				result += this.values[this.values.length-1]+"})";
			}
			else
				result = "(" + this.attrIndex + ", " + this.median + ")";

			return result;
		}

	}

	public class KNNList{

		private Neighbor head;
		private int size;
		private int maxSize;

		public KNNList(int k){
			this.size = 0;
			this.maxSize = k;
		}

		public boolean isEmpty(){
			return this.size() == 0;
		}

		public boolean isFull(){
			return this.size == this.maxSize;
		}

		public int size(){
			return this.size;
		}

		public void add(float[] newValues, float distToObs){
			//list is not full
			if(!this.isFull()){
				Neighbor newNeighbor = new Neighbor(newValues,distToObs);
				Neighbor nextNeighbor = this.head;

				//list is empty
				if(nextNeighbor == null){
					this.head = newNeighbor;
				}

				//list contains atleast 1 neighbor
				else{
					Neighbor prevNeighbor = null;
					while(nextNeighbor != null && distToObs < nextNeighbor.getDistanceToObs()){
						prevNeighbor = nextNeighbor;
						nextNeighbor = nextNeighbor.getNext();
					}					

					//newNeighbor is the new head
					if(prevNeighbor == null){
						newNeighbor.setNext(this.head);
						this.head = newNeighbor;
					}

					//reached end of list
					else if(nextNeighbor == null)
						prevNeighbor.setNext(newNeighbor);

					//newNeighbor lies between two other neighbors
					else{
						newNeighbor.setNext(nextNeighbor);
						prevNeighbor.setNext(newNeighbor);					
					}
				}
				this.size++;
			}		

			//list is full
			else if(distToObs < this.head.getDistanceToObs()){
				Neighbor newNeighbor = new Neighbor(newValues, distToObs);
				Neighbor nextNeighbor = this.head;

				Neighbor prevNeighbor = null;
				while(nextNeighbor != null && distToObs < nextNeighbor.getDistanceToObs()){
					prevNeighbor = nextNeighbor;
					nextNeighbor = nextNeighbor.getNext();
				}	

				newNeighbor.setNext(nextNeighbor);
				prevNeighbor.setNext(newNeighbor);
				this.head = this.head.getNext();

			}
		}

		public ArrayList<float[]> getNeighbors(){
			ArrayList<float[]> neighbors = new ArrayList<float[]>(this.maxSize);
			Neighbor currNeighbor = this.head;
			for(int i=0; i < this.size(); i++){
				neighbors.add(currNeighbor.getValues());
				currNeighbor = currNeighbor.getNext();
			}
			return neighbors;
		}

		public float getKthDistanceToObs(){
			return this.head.getDistanceToObs();
		}


		private class Neighbor{

			private float[] values;
			private Neighbor next;
			private float distanceToObs;

			public Neighbor(float[] values, float distanceToObs){
				this.values = values;
				this.distanceToObs = distanceToObs;
			}

			public Neighbor getNext(){
				return this.next;
			}

			public void setNext(Neighbor next){
				this.next = next;
			}			

			public float getDistanceToObs(){
				return this.distanceToObs;
			}

			public float[] getValues(){
				return this.values;
			}
		}
	}
}


