const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {

	/**
	 * Sets up all default properties
	 * @name constructor
	 * @param features - matrix of features
	 * @param labels - matrix of features
	 * @param options - override default options
	*/
	constructor(features, labels, options) {
		this.features = this.processFeatures(features); // all features
		this.labels = tf.tensor(labels); // all labels
		this.costHistory = []; // history of Cross Entrophy

		this.options = Object.assign({
			learningRate: 0.1,
			iterations: 1000,
			batchSize: 3,
			decisionBoundary: 0.5,
		}, options);

		this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]); // m and b values

	}

	/**
	 * multiply the features with the weights (m & b guesses)
	 * get the differences between the current guesses and the labels
	 * find the average slope between the features and the differences
	 * update the weights (m & b) to be the weights minus the slopes * learning rate
	 * @name gradientDescent
	 * @param features - batch of features
	 * @param labels - batch of labels
	*/

	gradientDescent(features, labels) {
		// matrix multiplication
		const currentGuesses = features.matMul(this.weights).softmax();
		const differences = currentGuesses.sub(labels);
		const slopes = features
			.transpose()
			.matMul(differences)
			.div(features.shape[0]);

		return this.weights.sub(slopes.mul(this.options.learningRate))
	}

	/**
	 * Initializes the training
	 * loop through each allowed iteration
	 * loop though each batch of features and run gradientDecent
	 * record the Mean Square Error
	 * update the learning rate
	 * @name train
	*/

	train() {
		const batchQuantity = Math.floor(
			this.features.shape[0] / this.options.batchSize
		);
		for (let i = 0; i < this.options.iterations; i++) {
			for (let j = 0; j < batchQuantity; j++) {
				const { batchSize } = this.options;
				const startIndex = j * batchSize;

				this.weights = tf.tidy(() => {
					const featureSlice = this.features.slice(
						[startIndex, 0],
						[batchSize, -1]
					);
					const labelSlice = this.labels.slice(
						[startIndex, 0],
						[batchSize, -1]
					)
					return this.gradientDescent(featureSlice, labelSlice);
				});

			}
			this.recordCost();
			this.updateLearningRate();
		}
	}

	/**
	  * process the features, and multiply by weights
		* run the softmax method on the outcome
		* .argMax get the largest value alone the horizontal axis
		* @name predict
		* @param observations - Array of array of observations
	*/
	predict(observations) {
		return this.processFeatures(observations)
			.matMul(this.weights)
			.softmax()
			.argMax(1);
	}

	/**
	 * Get the probability
	 * add them all together
	 * calculate the % we got correct
	 * @name test
	 * @param testFeatures
	 * @param testLabels
	 * @return {Number}
	*/
	test(testFeatures, testLabels) {
		const predictions = this.predict(testFeatures);
		testLabels = tf.tensor(testLabels).argMax(1);

		const incorrect = predictions.notEqual(testLabels).sum().get();

		return (predictions.shape[0] - incorrect) / predictions.shape[0];
	}

	/**
	  * Process a feature set of data
		* convert into a tensor
		* standardize the features
		* lastly, create a ones matrix with the number of rows of the feature set and concat with with features
		* @name processFeatures
		* @param features {Array}
		* @return {Tensor}
	*/
	processFeatures(features) {
		features = tf.tensor(features);

		features = this.standardize(features);
		features = tf.ones([features.shape[0], 1]).concat(features, 1);

		return features;
	}

	/**
	 * If the mean and variance already exist, run the standardization on the existing mean and variance
	 * If not, get the mean and variance by using tf.moments on features
	 * Fix the div by 0 NaN issue by creating the filler
	 * set the mean and variance props
	 * then run the standardization on the updated mean and variance
	 * @name standardize
	 * @param features
	*/
	standardize(features) {
		if(this.mean && this.variance) {
			return features.sub(this.mean).div(this.variance.pow(0.5));
		}

		const { mean, variance } = tf.moments(features, 0);
		const filler = variance.cast('bool').logicalNot().cast('float32'); // fixes 0s

		this.mean = mean;
		this.variance = variance.add(filler);

		return features.sub(this.mean).div(this.variance.pow(0.5));
	}

	/**
	  * Tracks the Cross Entrophy Error
		* get the guesses by multiplying the features by the weights and running softmax
		* multiply the labels by the the log of the guesses
		* multiply the labels by -1, add 1, and multiply by log of guesses * -1 + 1
		* add termOne and termTwo, divide by the number of features, * -1
		* put the new value at the beginning of the costHistory
		* @name recordCost
	*/
	recordCost() {
		const cost = tf.tidy(() => {
			const guesses = this.features.matMul(this.weights).softmax();
			const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
			const termTwo = this.labels
				.mul(-1)
				.add(1)
				.transpose()
				.matMul(
					guesses
						.mul(-1)
						.add(1)
						.add(1e-7) // 1 x 10 ^ -7 (makes sure we never take the log(0) (adding a very small number))
						.log()
				);

			return termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);
		});

		this.costHistory.unshift(cost);
	}

	/**
	 * Dynamically updates the learning rate
	 * if the mean square error history doesn't have enough history, skip
	 * if the last MSE is greater than the second to last MSE ---> divide the learning rate by half (it's way too big)
	 * else increase the learning rate by 5% ( make it slightly more in the right direction)
	 * @name updateLearningRate
	*/
	updateLearningRate() {
		if (this.costHistory.length < 2) return;
		if (this.costHistory[0] > this.costHistory[1]) {
			this.options.learningRate /= 2;
		} else {
			this.options.learningRate *= 1.05;
		}
	}

}

module.exports = LogisticRegression;
