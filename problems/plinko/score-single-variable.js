/*
 * Adjust the parameters of the analysis (k)
 * Add more features to explain the analysis
 * Change the prediction point
 * Accept that maybe there isn't a good correlation
*/

/**
  * datapoints
	* @name outputs
*/
const outputs = [];


/**
  * add ball drop data to the outputs
	* @name onScoreUpdate
	* @param dropPosition
	* @param bounciness
	* @param size
	* @param bucketLabel
*/
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

/**
  * size of sample we take from sorted data
	* @name k
*/
// const k = 3;

/**
 * Split the dataset into the test set and the training set
 * Loop through a set of "k" values to see which one works best
 * filter each test by comparing the knn results from the straining set vs testPoint and the actual result
 * get the size of the results
 * divide the size of accurate predictions by the testSetSize to get accuracy
 * log the percent correctness
 * @name runAnalysis
*/
function runAnalysis() {
	const testSetSize = 150;
	const [testSet, trainingSet] = splitDataset(outputs, testSetSize);

	// let numberCorrect = 0;
	// for (let i = 0; i < testSet.length; i++) {
	// 	const bucket = knn(trainingSet, testSet[i][0]);
	// 	if (bucket === testSet[i][3]) numberCorrect++;
	// 	console.log('predicted bucket: ', bucket, ' actual bucket: ', testSet[i][3]); // predicted bucket vs actual bucket
	// }
	// console.log('Correct ', (numberCorrect / testSetSize)*100, '% of the time.');

	_.range(7, 12).forEach(k => {
		const accuracy = _.chain(testSet)
		  .filter(testPoint => knn(trainingSet, testPoint[0], k) === testPoint[3])
			.size()
			.divide(testSetSize)
			.value();
			console.log('It is accurate ', accuracy*100, '% of the time using a k value of ', k);
	});

}

/**
	* map through all the elements and modify data point to distance from prediction point and remove unneeded data
	* sortBy the first index, which is the closeness that current row prediction is to the prediction point
	* slice the sorted list to only include the first "k"
	* countBy create a map that counts the occurences of each value (which bucket that occurence went into)
	* toPairs turn those key value pairs into an array of [key, value]
	* sortBy second index, so sort by the number of occurences
	* last get the last index, which will have the most occurences
	* first get the first index in the last arry from above which will be the value of the most occured bucket
	* turn it into a number
	@name knn
	@param data
	@param point
*/
function knn(data, point, k) {
	return _
		.chain(data)
	  .map(row => [ distance(row[0], point), row[3] ] )
	  .sortBy(row => row[0])
	  .slice(0, k)
	  .countBy(row => row[1])
	  .toPairs()
	  .sortBy(row => row[1])
	  .last()
	  .first()
	  .parseInt()
	  .value();
}

/**
  * get the distance from prediction point
  * @name distance
	* @param point datapoint
*/
function distance(pointA, pointB) {
  return Math.abs(pointA - pointB);
}

/**
  * gets a test set and a training set
  * @name splitDataset
	* @param data dataset
	* @param testCount number of test cases we want to try out
*/
function splitDataset(data, testCount) {
	const shuffled = _.shuffle(data);
	const testSet = _.slice(shuffled, 0, testCount);
	const trainingSet = _.slice(shuffled, testCount);
	return [testSet, trainingSet];
}
