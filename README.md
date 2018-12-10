## Machine Learning

Notes from this class on Udemy. [here](https://www.udemy.com/machine-learning-with-javascript/learn/v4/overview)


## Tool Notes

### Lodash

Pros:

* Methods for almost everything we need
* Great API design
* Skills are transferrable to other JS projects

Cons:

* Very slow (relatively)
* Not 'numbers' focused
* Some things are awkward

### Tensorflow JS

Pros:

* Similar API to Lodash
* Extremely fast for numeric calculations
* Has a 'low level' linear algebra API + higher level API for ML
* Similar api to Numpy - popular Python numerical lib

Cons:

* Still in active development


## Steps for machine learning

If something changes then something else might change.

* Identify data that is relevant to the problem
* Assemble a set of data related to the problem you're trying to solve
* Decide on the type of output you are predicting
* Based on the type of output, pick an algorithm that will determine a correlation between your 'features' and 'labels'
* Use model generated by algorithm to make a prediction

Features are categories of data points that affect the value of a 'label'

### Identifying Data Needed (Identifying Data)

Look at the problem you're trying to solve and decide what type of data you need.

### Getting data (Assembling Data)

You may need to search multiple sources for data. Often data doesn't come in cleaned up packages so you will need to create the data on your own.

### Type of Data We Might Try to Predict (Type of Output)

* `Classification` - The value of our labels belong to a discrete set.
  * Based on how many hours a student studied for an exam, did they PASS or FAIL (PASS or FAIL)
  * Based on the content of this email, is it SPAM or NOT SPAM (SPAM or NOT SPAM)
  * Based on where a football player shoots from, are they likely to SCORE or NOT SCORE (SCORE or NOT SCORE)
* `Regression` - The value of our labels belong to a continuous set.
  * Based on the year, make, and model of a car, what is it's value? ($0 - $50k)
  * Based on an individual's daily calorie intake and minutes spent exercising, what is their weight? (80lb - 400lb)
  * Based on the height of this pine tree, what is it's age? (0 years - 500 years)


### Picking an Algorithm

Select an algorithm that fits the data selection and desired output.

### Use model to predict

Use your decided algorithm and dataset to try to make a prediction.


## Plinko

`/projects/plinko/` Given some data about where a ball is dropped from, can we predict what bucket it will end up in?

* Where are we dropping the ball
* Range of Ball Bounciness
* Range of Ball Size (how large the ball is)

This problem will have 3 features and one label.

* Features (Changing one of these)
  * Drop Position
  * Ball Bounciness
  * Ball Size
* Labels (Will probably change this)
  * Buckets a ball lands in

### Approaches

Array of Objects Approach - can get confusing when parsing

```
  [
    { dropPosition: 300, bounciness: 0.4, ballSize: 16, bucket: 4 },
    { dropPosition: 300, bounciness: 0.4, ballSize: 16, bucket: 4 },
    { dropPosition: 300, bounciness: 0.4, ballSize: 16, bucket: 4 },
    { dropPosition: 300, bounciness: 0.4, ballSize: 16, bucket: 4 }
  ]
```

Array of Array Approach - Need to keep track of what index means what

```
  [
    [300, 0.4, 16, 4],
    [350, 0.4, 25, 5],
    [416, 0.4, 16, 4],
    [722, 0.4, 16, 7]
  ]

```

### Type of Output

Since we are trying to determine which bucking the ball will fall into, this would be a `Classification` since there are a finite amount of buckets.

### Algorithm Selection

`K-Nearest Neighbor (knn)` - "Birds of a feather flock together"

`k` = number of records to use once sorted (this will be tweaked depending on your solution)

* `Features` - Data being used to predict an outcome
* `Labels` - Data you're trying to predict the outcome of
* `Test Data` - Data you are using to determine accuracy
* `Training Data` - Data you are using to as your main dataset
* `Feature Normalization` - Putting data on a scale of 0 - 1 instead of any value (find min max of set of data and scale it proportionally)
* `Common Data Structures` - Arrays of arrays
* `Feature Selection` - Selecting `features` based on their accuracy and ability to predict an outcome


#### Examples

##### Single Variable

Working Example Code `./problems/plinko/score-single-variable.js`

Which Bucket will a ball go into if dropped at 300px.

* Drop a ball a bunch of times around the board, record which bucket it goes into
* For each observation, subtract drop point from 300px, take absolute value (this determines how close it is to the 300px drop we are trying to guess)
* Sort the results from least to greatest
* Look at the 'k' top records. What is the most common bucket? (k is a number of records you are going to use to find the most common bucket)
* Whichever bucket came up most frequently is the one ours will probably go into

__

* Create a dataset, here the data array is `[dropPosition, bounciness, size, bucketLabel]`


```
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}
```

* Get the distance between two pieces of the same data

```
function distance(pointA, pointB) {
  return Math.abs(pointA - pointB);
}
```

* Set the `testSetSize` to be how many test example you want to use (this needs to be refined depending on the dataset)
* Split the dataset into the `testSet` and the `trainingSet`
  * `trainingSet` - The set of data you're going to sort and organize to come up with a prediction
  *  `testSet` - The set of data you're going to use as a case to test the results from the `trainingSet`
* Loop through a set of `k` ranges (needs to be refined by developer)
* filter each set of data in the `testSet` run through the `knn` method passing the `trainingSet` and the `dropPosition` of that piece of the `testSet`, then compare the resulting bucket to the actual bucket of the current piece of the `testSet` (see if the the predicted results are the actual results)
* Get the size of the filtered `testSet`
* Divide by the size of the of your testSet to get the % accuracy

```
function runAnalysis() {
	const testSetSize = 150;
	const [testSet, trainingSet] = splitDataset(outputs, testSetSize);

	_.range(7, 12).forEach(k => {
		const accuracy = _.chain(testSet)
		  .filter(testPoint => knn(trainingSet, testPoint[0], k) === testPoint[3])
			.size()
			.divide(testSetSize)
			.value();
			console.log('It is accurate ', accuracy*100, '% of the time using a k value of ', k);
	});

```

Get the absolute distance between the training piece of data (dropPoint) and these test set result (bucket), sorting by relevance/most similar (lowest to highest), getting the `k` top most similar matches, count the occurrences of certain results (bucket) for most similar distances. Return.

* Accept `data`, `point`, `k`
  * `data` - set of data (training set)
  * `point` - point of data you want to test for (variable you're testing for)
  * `k` - size of sample to take from sorted `data`
* map through all training set data and get the absolute distance between the training set's dropPosition and the test sets dropPosition. Add it to an array and also add the training set data's resulting bucket `['distance dropPosition', 'result from this training set']`
* sort by first element in the resulting array (this will be lowest the greatest, aka most similar to least similar by dropPosition)
* Slice the sorted array by `k` value, getting the first `k` most similar matches
* countBy to make an object map that counts the occurrences of a certain bucket/result for the `k` most similar training sets
* toPairs - covert key - value pair to arrays
* sort the arrays by the the value ([1] index) (sorting by the most occurrences)
* get the last array (one with the highest count)
* get the first element in the above array (most common resulting bucket)
* turn into a number

```
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
```

##### Multiple Variables

Which Bucket will a ball go into if dropped at 300px and the bounciness was 0.5.

* Drop a ball a bunch of times around the board, record which bucket it goes into
* For each observation, find the distance from the observation to prediction point of (300, 0.5)
* Sort the results from least to greatest
* Look at the `k` top records. What is the most common bucket?
* Whichever bucket came up most frequently is the one ours will probably go into

To calculate multiple variable we essentially use the Pythagorean Theorem `C^2 = A^2 + B^2` where the hypotenuse will be the true distance between two points. In the below, we would be calculating the distance between the blue point and the green point using the Pythagorean Theorem.

<img src="images/graph.png"/>


If you have more than 2 variable you can imagine this takes place on a 3D graph, using the Pythagorean Theorem is still quite simple.

<img src="images/graph2.png"/>


##### Normalization

You will want to normalize or standardize you data so they fall on the same scale of 0 - 1.

`(NORMALIZED DATASET) = (FeatureValue - minOfFeatureValues) / (maxOfFeatureValues - minOfFeatureValues)`

<img src="images/normalization.png"/>


##### Feature Selection

You may consider if the change to a certain feature makes predictable changes to the output while changes of may not make predictable changes. In the example of the ball drop changes to the *dropPosition* make far more predictable change to the outcome compared to changes to the ball bounciness. Selecting which features to include in an analysis is called `Feature Selection`. You may decide to remove a `feature` from the analysis if it is too unpredictable or throws off the results.

In `knn` analysis, you could run the analysis with each feature to determine the most important features (one's with more accuracy). Also, you're looking for a % accuracy significantly above just guessing.
__
