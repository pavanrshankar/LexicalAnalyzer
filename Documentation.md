Application Domain:

Our language is specific to Classification Algorithms. Classification is a major Machine Learning methodology that involves 
learning from a given data and deriving some meaning from new data. The classification algorithms are not easy and intuitive 
to implement. There are different ways in which every algorithm is modeled and different working paradigm behind each of those. Providing a standard for writing such programs is crucial but at the same time customizability must be offered to the 
machine learning researcher/programmer. We chose 4 popular classification algorithms Neural Networks, Naïve Bayes, 
Linear regression with gradient descent and K Nearest neighbors and made a language that enables easy implementation 
of these algorithms but at the same time allows low level modifications.

Some problems in the domain that can be solved by a language:

1. Mathematical functions such as sigma, exponent, sigmoidal etc.

2. Matrix operations – Data Science requires matrix operations, if arrays can be worked on by the basic operands 
   like +, -, %, * and so on, it will provide a certain degree of freedom to the programmer.
   
3. Containers for working with data - Data science requires working with image data for image processing, csv, xls files etc. 
   If the language can provide containers for such data, it will be much easier for performing several tasks 
   e.g. If image file is loaded into a container that allows pixel by pixel traversal then it can save time for image 
   processing.
   
4. Flexible range based iterators for iterating through data/ nodes of neural network/ cost function vectors etc.

5. Automatic training, testing and classification support

6. Some condition based loops like till error < threshold or till cost converges to 0.

7. Frameworks for models that quick manipulation of model parameters.

8. Database connection – Data science requires working with large amount of data and often we want to select certain features 
   of given data set for a specific purpose. Databases store data efficiently and allow complex selection and join operations.
   If the programming language can allow working with databases at a very fundamental level then it would increase scope 
   for seamless interaction with data.

9. Non – blocking assignment and non – blocking operations: Often it is required in Data Science application to simultaneously 
   update data. Non – blocking assignments can allow writing of very intuitive and precise code for this purpose.
   
Programming features of language:

1. Language provides easy readability of code due to high level abstractions of classification models and simplified 
   mathematical operations. E.g. For creation of KNN classifier, we can specify
                                          
                                          classificationModel<KNN> knn;
                                          
2. Language also provides easy writability of code. E.g. array addition, subtraction and multiplication are simplified. 
                                
                                  Arr1[] + Arr2[] adds all the elements at same index. 
                                  
   Easy file access is provided for training and testing like:

                              vector<int> irisresults = classifyFromFile(„iris.txt‟)
                              
3. Language conforms to size of int as 4 bytes, size of double as 8 bytes as specified by IEEE. Implicit type conversion of 
   int to double is allowed. Explicit type conversion of int to double and double to int allowed. Apart from this all other
   type conversion is illegal.
   
Helpful features of our language (with examples in code):

1. Array operations: Addition, subtraction, multiplication, increment, &, | etc. operations that are supported on integers 
   will be supported on integer arrays. Such operations will be applied element wise to the arrays. For two arrays, 
   it will be applied elements with equal indexes in respective arrays but will be supported on if array sizes are equal. 
   This will be useful for data science as we need to apply many operations on matrices. 
   E.g. //operations on arrays
   
                              int arr1[10][10], arr2[10][10];
                              arr1[][]++; //add 1 to all elements of arr1
                              arr2[][] – arr1[][]; /* subtract elements of arr1 from arr2 at same index */
                              
2. Mathematical functions:
   a. Sigma function: This is used for summation. Syntax:

                           sigma(iterating variable, start value, end value){
                              //formula to be summed
                              heights[iterating variable];
                           }
                           
   b. Sigmoid function: This function accepts a variable and return sigmoidal value
      
                           vector<int> sigmoid_values;
                           for i in range (1,10){
                              sigmoid_values.insert(4*sigmoid(i) +1);
                           }
                           
   c. Exponent function: Return e^argument.
   
                           printf(exp(10));
                           
3. Neural Networks:
   i. Model attributes:
       Layers vector (add layer) – can be iterated using for (shown in example)
       Learning rate
       Input layer, Output layer

   ii. Layer attributes:
       Eg. Layers[layer name][node number]
       Node vector – can be iterated using for (shown in example)
       addNode, numNodes
       layerName - string
      
   iii. Node attributes:   
       Weights- dictionary {<layer name, node number> : weight, <layer name, node number> : weight}
       Step function
       Threshold
      
   iv. TrainModel, TestModel, Classify
      Code:
      
                           classificationModel<ANN> myNet;
                           myNet.inputLayer=‟input‟;
                           myNet.outputLayer=‟output‟;
                           myNet.learningRate=0.4;
                           myNet.layer['input'].addNodes(4); //adds 4 nodes to layer named input
                           myNet.layer['output'].addNodes(1);
                           myNet.layer['hidden'].addNodes(2);
                           //layer['foo'] will create a new layer named foo if not present
                           myNet.layer['hidden'][1].weights = { <‟input‟,1>: 0.5, <‟input‟,2>: 0.5 };
                           myNet.layer['hidden'][2].weights = { <‟input‟,3>: 0.5, <‟input‟,4>: 0.5 };
                           myNet.layer['output'][1].weights = { <‟hidden‟,1>: 0.5, <‟hidden‟,2>: 0.5 };
                           
                           for level in myNet.layer{
                              for node in myNet.layer(level){
                                 myNet.layer['hidden'][i].stepFunction = "sigmoid";
                                 myNet.layer['hidden'][i].threshold = 0.46;
                              }
                           }
                           
                           myNet.trainModel("neuralTrainingData.csv");
                           testResults<ANN> annres= myNet.testModel("neuralTestingData.csv");
                           printf('ANN model test results')
                           vector<int> sample = {3.5, 2.47452, 0.004112, 124}
                           int result = myNet.classify(sample);
                           vector<int> irisResults = myNet.classifyFromFile("irisData.csv");
                           
4. Regression using Gradient Descent
   i. Model attributes:
       Hypothesis coefficient vector
       Cost function, Error
       X and Y vectors – the input and target vectors
      Code:

               classificationModel<RGD> regression;
               regression.hypothesis.size = 3; // hypothesis of the form a0 + a1x + a2x2
               for dim in regression.hypothesis{
                  dim=1; //hypothesis initialized to 1 + x + x2
               }
               regression.costFunction = sigma(i,1,3){ pow(regression.hypothesis[i] * regression.X[i] – regression.Y[i],2) };
               untilConverge(regression.error = 0.1){
                  regression.trainModel("regressionTrainingData.txt");
               }
               testResults<RGD> testing= regression.testModel('regressionTestingData.txt');
               print(testing);
               vector<double> result = regression.classifyFromFile('propertyValues.csv');
               printf('Property Value results:\n' + result);

5. KNN
   i. Model functions:
       createModel(string filename)
       testResults<knn> testModel(string filename, int k_value) – testResults.error is updated
       vector<string> classify(string filename)
       setDistanceParameter(double parameter) – parameter of 1 corresponds to Hamiltonian distance, 2 corresponds to                Euclidean distance and so on.
       vector<string> confidenceCalculation(vector point) – vector returned contains class names in decreasing order of             confidence with respect to data point calculated by default approach. For custom confidence calculation, set the             confidence logic code in KNN model is as follows:

               knn_model.confidenceCalculation(vector data_point) = {
               // logic for Confidence Calculation
               }
               
   ii. for Construct:
              
               for point in km.nearestNeighbours(new_point){ //Statements }
               
       K value can be selected by restricting majority voting to selected points which are within a distance from the data          point.
       
   iii. Model attributes:
       Distance parameter
      Code:
      
               classificationModel<KNN> knn;
               knn.createModel('iris.txt');
               knn.setDistanceParameter(3);
               int threshold = 4;
               vector<int> new_point = (2,3,6,1,0);
               vector<string> class_names;
               for point in knn.nearestNeighbours(new_point){
                  /*dist() is used for distance calculation between 2 vectors based on distance parameter */
                  if(knn.dist(new_point, point) < threshold){
                     class_names.add(point[-1]);
                  }
                  else{
                     break;
                  }
               }
               //majority returns the terms with highest frequency
               class_names = majority(class_names);
               if(size(class_names) == 1){
                  printf(“Class name: ”+class_names[0]);
               }
               else{
               /* Custom confidence calculation by choosing the class which contains a point closest to data point */
                  knn.confidenceCalculation(vector data_point) = {
                     for point in knn.nearestNeighbours(new_point){
                        if(point[-1] in class_names){
                           break;
                        }
                     }
                     return point[-1];
                  }
               }
               
6. Naïve Bayes
i. Model Attributes:
 priorProbabilities – dictionary containing classname as key and prior Probabilities as value
ii. Model Functions:
 trainModel(string filename, vector<string> classnames)
 testResults<naiveBayes> testModel(filename) - testResults.error updated.
 vector<string> classify(filename)
iii. Constructs:
 Probability calculations are dependent on whether attribute is discrete or continuous. Class conditional probability calculation for discrete attributes is done in a frequentist approach by counting the number of occurrences. For continuous attributes Gaussian distribution is used.
 Custom class conditional probability is set as follows
naïveBayesModel.defaultProbabilityCalculationDiscrete (vector<string> attribute, Boolean change_value)
[‘all’] is specified for all the attributes to follow new probability calculations else specified vector of strings follow new probability calculations.
naïveBayesModel.discreteProbabilityCalculation(int desirable_outcomes, int total_outcomes) = {
//new probability estimation
}
Code:
Consider the training data set with input fields Wind speed, Power output, Generator Winding Temperature and output field Wind Turbine status as specified in wind.txt. Unclassified data points are specified in classification.txt file. Usage of Naïve Bayes classifier is as follows:
classificationModel<naiveBayes> nbc;
nbc.trainModel(„wind.txt‟);
nbc.defaultProbabilityCalculationDiscrete(„Power output‟, false);
nbc.discreteProbabilityCalculations(int desirable_outcomes, int total_outcomes) = {
int m = 2; //setting m-estimate parameter
double p = (desirable_outcomes / total_outcomes);
return (desirable_outcomes + m*p) / (total_outcomes + m);
}
result = nbc.classify(„testing.txt‟);
printf(“Class of result is: ”+result);
7. Non-blocking assignments & assignment blocks: Non-blocking assignment like the ‘<=’ in Verilog can be helpful for simultaneous update of multiple items without regard to order or dependence upon on each other. Once example of where it will be useful is when we update parameters of cost-function using gradient descent.
//Non-blocking assignment
a := b;
b := a; //this will swap a and b
nonBlocking{
//j is array of coefficients of cost function
j[0] = j[0] – diff(j[],0);
j[1] = j[1] – diff(j[],1);
j[2] = j[2] - diff(j[],2);
j[3] = j[3] – diff(j[],3);
}//all four updates happen in a non-blocking manner
8. dataContainer data type: This is a data type that will act as a container for data used for data science tasks. It is similar to how C++ defines containers like Vector, Stack and Queue of types int, char etc. e. g. Vector<int> or Stack<char>. The types accepted by the data container will be image, csv, xls. Type image will allow pixel by pixel traversal, useful for image processing purposes. Type csv and xls can be used for reading in data from csv and excel files and used for traversing through data points, used for getting field names and manipulating data at a lower level.
//creating a data container of image type
dataContainer<image> img= loadImage(“~/pictures/iris.jpg”);
9. Database functions: Connection to database can be established easily and queries run on it directly with simple syntax. Currently, with the rise of big data, there is increasing need for running machine learning and AI algorithms on data easily selected through joins/views from an efficient database system. This feature will decrease hassle of any programmer trying to work with both data from a database and ML algorithms.
//creating a database variable with connection details
database db = connect(“user_name”,”sales_db”,”localhost”);
int max = db(“select max(sales) from shoe_sales”);
The above code creates a link to database named sales_db on server localhost. User name has to be specified to verify access to database. Pass the SQL query to database link to execute it.
10. HTTP requests: Using get, put, post and delete HTTP requests, we can easily interact with data on servers. Very useful for interacting with cloud based data storage services. This removes the limitation of a data science programmer to interact and work with data on a local machine.
put(“http::/server/script/resources?query”);

List of tokens types:

1. Keywords (list of keywords mentioned in next page)

2. Operators (++ , -- , - , + , / , * , ^ , | , & , || , && , ! , = , == , < , > , ?, :=)

3. Delimiters (( , ) , { , } , [ , ] , ; )

4. Special symbols (//, /*, */, :)

5. Identifiers – case sensitive words starting with letter followed by letter/ number / _

List of keywords:

1. dataContainer<generic type> can be  image, audio, int, bool, double, string, xls, csv, txt

2. model, trainModel, testModel, saveModelToFile, loadModelFromFile, classificationModel, testResults

3. database, connect

4. nonBlocking

5. get, put, post, delete

6. for, while, do, iterator, range, untilConverge, in

7. switch, case, break, continue, if, else, return

8. int, double, bool, string, struct, void

9. from, import

10. true, false

11. dict -> {set of all key: value pairs with distinct keys}

12. stack, queue, tree, list, vector, set

13. ANN, RGD, KNN, naiveBayes

14. Sigma, sigmoid, exp
