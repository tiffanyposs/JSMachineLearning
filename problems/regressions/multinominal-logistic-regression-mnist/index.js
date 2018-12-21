require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

function loadData() {
	// the the mnist data
	const mnistData = mnist.training(0, 60000);

	// get the image pixel data and turn it into a single array
	const features = mnistData.images.values.map(image => _.flatMap(image));

	// encode the labels based on which number they actually are
	const encodedLabels = mnistData.labels.values.map(label => {
		const row = new Array(10).fill(0);
		row[label] = 1;
		return row;
	});

	return { features, labels: encodedLabels }
}

const { features, labels } = loadData();

// set up class
const regression = new LogisticRegression(features, labels, {
	learningRate: 1,
	iterations: 20,
	batchSize: 100,
});

// train the class
regression.train();
debugger;

// setup test data
const testMnistData = mnist.testing(0, 1000);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
	const row = new Array(10).fill(0);
	row[label] = 1;
	return row;
});

// run .test for accuracy
const accuracy = regression.test(testFeatures, testEncodedLabels);

console.log('Accuracy is: ', accuracy);
