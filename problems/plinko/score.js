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
 * Set the testSetSize
 * set a k value
 * loop through each feature index
 * get the data used for each feature [feature, label]
 * Split the dataset into the test set and the training set
 * filter each test by comparing the knn results from the straining set vs testPoint and the actual result
 * get the size of the results
 * divide the size of accurate predictions by the testSetSize to get accuracy
 * log the percent correctness
 * @name runAnalysis
*/
function runAnalysis() {
	const testSetSize = 100;
	const k = 10;

	_.range(0, 3).forEach(feature => {
		const data = _.map(outputs, row => [row[feature], _.last(row)])
		const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);
		const accuracy = _.chain(testSet)
		  .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint))
			.size()
			.divide(testSetSize)
			.value();
			console.log('It is accurate ', accuracy*100, '% of the time using a feature value of ', feature);
	});
}

/**
 * Split the dataset into the test set and the training set
 * Loop through a set of "k" values to see which one works best
 * filter each test by comparing the knn results from the straining set vs testPoint and the actual result
 * get the size of the results
 * divide the size of accurate predictions by the testSetSize to get accuracy
 * log the percent correctness
 * @name runAnalysis
*/

// function runAnalysis() {
// 	const testSetSize = 100;
// 	const [testSet, trainingSet] = splitDataset(minMax(outputs, 3), testSetSize);
// 	_.range(1, 20).forEach(k => {
// 		const accuracy = _.chain(testSet)
// 		  .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3])
// 			.size()
// 			.divide(testSetSize)
// 			.value();
// 			console.log('It is accurate ', accuracy*100, '% of the time using a k value of ', k);
// 	});
// }

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
	@param data [dropPosition, bounciness, size, bucketLabel]
	@param point [dropPosition, bounciness, size]  (without bucketLabel)
*/
function knn(data, point, k) {
	return _
		.chain(data)
	  .map(row => {
			return [
				distance(_.initial(row), point),
				_.last(row)
			]
		})
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
  * take a series of data you want to test [300, .5] and [350, .55]
	* zip together two sets of data matching the same data category [[300, 350], [.5, .55]]
	* map through the zipped data and subtract one from the other and square it [2500, 0.0025000000000000044]
	* add them all together
	* get the square root of these
	*
  * @name distance
	* @param pointA datapoint Array ex: [300, .5, 16]
	* @param pointB datapoint Array ex: [350, .55, 16]
*/
function distance(pointA, pointB) {
	return _.chain(pointA)
	  .zip(pointB)
	  .map(([a, b]) => (a - b) ** 2)
	  .sum()
	  .value() ** 0.5;
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

/**
  * clone that passed data
	* for each feature count loop through and extract that column
	* extract the min and max of that column/feature
	* then, for each row, update the current column with the formula that calculates normalization
  * @name minMax
	* @param data dataset
	* @param featureCount - how many features to update in an array of data
*/
function minMax(data, featureCount) {
	const clonedData = _.cloneDeep(data);
	for (let i = 0; i < featureCount; i++) {
		const column = clonedData.map(row => row[i]);
		const min = _.min(column);
		const max = _.max(column);
		for (let j = 0; j < clonedData.length; j++) {
			clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
		}
	}

	return clonedData;
}
