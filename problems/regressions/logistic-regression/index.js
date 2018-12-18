require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');

const loadCSV = require('../load-csv');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
	dataColumns: [
		'horsepower',
		'displacement',
		'weight'
	],
	labelColumns: [
		'passedemissions',
	],
	shuffle: true,
	splitTest: 50,
	converters: {
		passedemissions: (value) => {
			return value === 'TRUE' ? 1 : 0;
		}
	}
});

const regression = new LogisticRegression(features, labels, {
	learningRate: 0.5,
	iterations: 3,
	batchSize: 10,
	decisionBoundary: .6,
});

regression.train();

const test = regression.test(testFeatures, testLabels);

console.log(test); // 88% correct

regression.predict([
	[130, 307, 1.752], // 0.23 -> likely to fail ---> actual: false
	[95, 113, 1.186] // 0.90 -> likely to pass ---> actual: true
]).print();

// passedemissions,mpg,cylinders,displacement,horsepower,weight,acceleration,modelyear,carname
// FALSE,18,8,307,130,1.752,12,70,chevrolet chevelle malibu
// TRUE,24,4,113,95,1.186,15,70,toyota corona mark ii

plot({
	x: regression.costHistory.reverse()
})
