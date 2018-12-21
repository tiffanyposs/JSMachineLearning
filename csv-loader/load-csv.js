const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

function extractColumns(data, columnNames) {
	const headers = _.first(data);
	const indexes = _.map(columnNames, column => headers.indexOf(column));
	const extracted = _.map(data, row => _.pullAt(row, indexes));
	return extracted;
}

module.exports = function loadCSV(
	filename,
	{ converters = {}, dataColumns = [], labelColumns = [], shuffle = true, splitTest = false, shufflePhrase = 'phrase' }
) {
	let data = fs.readFileSync(filename, { encoding: 'utf-8' }); // read file
	data = data.split('\n').map(row => row.split(',')); // split into an array on the comma
	data = data.map(row => _.dropRightWhile(row, val => val === '')); // remove empty columns at end
  data = data.filter(row => row.length); // remove empty rows

	const headers = _.first(data); // extract header

	data = data.map((row, index) => {
		if (index === 0) {
			return row;
		}
		return row.map((element, index) => {
			if (converters[headers[index]]) {
				const converted = converters[headers[index]](element);
				return _.isNaN(converted) ? element : converted;
			}
			const result = parseFloat(element);
			return _.isNaN(result) ? element : result; // if it's a number return a converted number, if not return the original
		})
	});

	let labels = extractColumns(data, labelColumns);
	data = extractColumns(data, dataColumns);

	data.shift();
	labels.shift();

	if (shuffle) {
		data = shuffleSeed.shuffle(data, shufflePhrase);
		labels = shuffleSeed.shuffle(labels, shufflePhrase);
	}

	if (splitTest) {
		 const trainSize = _.isNumber(splitTest) ? splitTest : Math.floor(data.length / 2);
		 return {
			 features: data.slice(0, trainSize),
			 labels: labels.slice(0, trainSize),
			 testFeatures: data.slice(trainSize),
			 testLabels: labels.slice(trainSize),
		 }
	} else {
		return {features: data, labels };
	}
}

const { features, labels, testFeatures, testLabels } = loadCSV('data.csv', {
	dataColumns: ['height', 'value'],
	labelColumns: ['passed'],
	shuffle: true,
	splitTest: 2, //
	converters: {
		passed: val => val === 'TRUE'
	}
});

console.log(features);
console.log(labels);
console.log(testFeatures);
console.log(testLabels);
