require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');

const loadCSV = require('../load-csv');

// Given the horsepower, weight, and displacement of a vehicle, will it have high, medium, or low fuel efficiency?
// 0 - 15 (low), 15 - 30 (medium), 30+ - high
const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
	dataColumns: [
		'horsepower',
		'displacement',
		'weight'
	],
	labelColumns: [
		'mpg',
	],
	shuffle: true,
	splitTest: 50,
	converters: {
		mpg: (value) => {
			const mpg = parseFloat(value);
			if (mpg < 15) {
				return [1, 0, 0];
			} else if (mpg < 30) {
				return [0, 1, 0];
			} else {
				return [0, 0, 1];
			}
		}
	}
});

const regression = new LogisticRegression(features, _.flatMap(labels), {
	learningRate: 0.5,
	iterations: 100,
	batchSize: 10,
	// decisionBoundary: .6,
});

regression.train();

const test = regression.test(testFeatures, _.flatMap(testLabels));
console.log(test);
// regression.predict([
// 	[215, 440, 2.16], // 14 actual (low), [1, 0, 0] - "low" - correct
// 	[85, 112, 1.2875], // 31 actual (medium), [0, 1, 0] - 'medium' - correct
// 	[150, 200, 2.223] // , [1, 1, 0] -
// ]).print()

// passedemissions,mpg,cylinders,displacement,horsepower,weight,acceleration,modelyear,carname
// TRUE,31,4,112,85,1.2875,16.2,82,pontiac j2000 se hatchback

//
// const test = regression.test(testFeatures, testLabels);
//
// console.log(test); // 88% correct
//
// regression.predict([
// 	[130, 307, 1.752], // 0.23 -> likely to fail ---> actual: false
// 	[95, 113, 1.186] // 0.90 -> likely to pass ---> actual: true
// ]).print();
//
// // passedemissions,mpg,cylinders,displacement,horsepower,weight,acceleration,modelyear,carname
// // FALSE,18,8,307,130,1.752,12,70,chevrolet chevelle malibu
// // TRUE,24,4,113,95,1.186,15,70,toyota corona mark ii
//
// plot({
// 	x: regression.costHistory.reverse()
// })
