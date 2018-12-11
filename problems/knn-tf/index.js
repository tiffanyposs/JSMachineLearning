// tells it where to run the calculations
require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

const k = 10; // -1.4381119101191653 error

/*
 * get the mean and variance of all the features on the 0 axis
 * scale the predictionPoint using STANDARDIZATION formula
 * subtract the mean and div by the square root of variance to get the STANDARDIZATION formula
 * subtract prediction point
 * square it
 * sum each row
 * square root of each row
 * expandDims - expand each row into another dimension
 * concat labels onto new dims
 * unstack into tensors
 * sort from least to greatest
 * get the first k matches
 * reduce them and get the average
*/
function knn(features, labels, predictionPoint, k) {
	const { mean, variance } = tf.moments(features, 0);
	const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

	return features
	  .sub(mean)
		.div(variance.pow(0.5))
	  .sub(scaledPrediction)
		.pow(2)
	  .sum(1)
	  .pow(0.5)
	  .expandDims(1)
		.concat(labels, 1)
		.unstack()
	  .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
	  .slice(0, k)
		.reduce((acc, pair) => acc + pair.get(1), 0) / k;
}

// shuffle the data, have 10 test rows, get the data columns for features "lat" and "long", the labelColumns are 'price'
let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
	shuffle: true,
	splitTest: k,
	dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living', 'yr_built', 'bathrooms'], // -1.4381119101191653 error
	labelColumns: ['price'],
});

// convert to TensorFlow
features = tf.tensor(features);
labels = tf.tensor(labels);

// const k = 10; // -1.4381119101191653


const errs = [];
// loop through each of the test features
testFeatures.forEach((testPoint, i) => {
	const result = knn(features, labels, tf.tensor(testPoint), k);
	const err = ((testLabels[i][0] - result) / testLabels[i][0])*100;
	errs.push(err);
	console.log('Guess: ', result, testLabels[i][0], ', That is a ', err, ' difference.');
});

const averageErrs = errs.reduce((a, b) => {
  return a + b;
}) / errs.length;

console.log('Average Error: ', averageErrs);
