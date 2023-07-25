# deep-learning-challenge  Challenge Week 21 Richard Soos
# This read me file contains both instructions on operating the scripts
# and the analysis for Step 4 Criteria is presented at the end

# Use of Programs
# Purpose. A Venture Capitl firm Alphabet Soup provides funding to applicants.
# This project writes a machine learning neural network based on this data.
# This data is available from a hyperlink within the program, though a copy is 
# also available at the Starter Ciode subfolder.
# Data cleaning techniques have been used, including removing irrelevant columns,
# binning feature columns as groups and standardisation.
# To run the initial program, run script Start_Code.ipynb
# The output file is the model data itself, called charity_keras_tensorsaved_model.h5

# If you would like to read this file, run script Read_my_hdf5.ipynb
# If you would like to import the learning model, run reuploadmodel.ipynb

# Extension program
# My revised tuning program is AlphabetSoupCharity_Optimisation.ipynb
# This program outputs the model to AlphabetSoupCharity_Optimisation
# If you wish to read this file, you can run script Read_my_hdf5.ipynb 
# but first rename the hardcoded file to read within.
# If you wish to import this learning model, run reuploadmodel.ipynb
# but first change the name of the model file hardcoded within

Step 4 Criteria
# Our initial model gave these results for testing
 Loss: 0.5542104840278625, Accuracy: 0.7244315147399902
I was determined to make the model as simple, yet more accurate.
The resultant model is slower.

*)  Which variables are the target of the model?
A) Y is the 1-value list of target values. It is the column "IS_SUCCESSFUL"
   with a binary value of 0/1 for each row.
*)  Which variables are the features of your model?
A) As I use the pca method to create 3 pseudo columns, I cannot tell you with   
   certainty which. pca will null.ify some columns contributions.
   All columns except for y, as per the 1st part of the assignment, are available.
    The variance of these pca columns was  
     array([1.00000000e+00, 1.08525466e-16, 5.83194850e-17]), suggesting that the 1st column represented 100% of the variance. 
   I used the same model as our model in step 2, but with 1 PCA column and the loss and accuracy were poor. I then set the PCA columns to 3 and used our original neural model.  The results were great. This is hard to explain, that one pca pseudo column supposedly accounted for 100% of the variability, yet I needed 3 to improve the accuracy by more than another 20% 

*) What variables() should be removed from the input data because they are neither targets
   nor features?
A) Misleading values like serial number must be removed as their numerical value will 
   confuse the model (although we did not have this for this assignment). 
   Columns which are essentially row identifiers like an index column or name of business
   should also be removed.

*) How many neurons, layers and activation functions did you select for your neural   
   network model, and why?

A) I did Not want to increase layers or neurons as I wanted to keep the model simple.
   I did swap the neuron count , from:
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 6     
         to:
         hidden_nodes_layer1 = 6
         hidden_nodes_layer2 = 8

I did this because the model is a forward directional model and intuitively I thought that complexity should increase through the modelling in later layers.
I kept the same 2 hidden layers. I changed one of the activation functions so that
layer2 now uses tanh instead of relu. Relu only returns max valkue if +ve, otherwise 0.
I wanted to use the hyperbolic tangent to give a range of values [-1,1]

*) Were you able to achieve the target model performace?
A) This easily hit our target, and so our work was done.


Loss: 0.16247129440307617, Accuracy: 0.9399417042732239
The training data was very near.
Epoch 100: Loss=0.1641, Accuracy=0.9387

*) What steps did you take in your attempts to increase model performance?
A) Changing epoch values to a higher value was of little to no gain, reducing it eroded the model's results very quickly.

Explicitly setting Regularization Hyperfunctions did Not help
Even though I used methods like this:
#nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, #activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="tanh", #kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

Setting the correct node count per layer, the correct activation function and using pca method to create limited pseudo columns were the extent of the work I did.

Later I released I code have simply changed the sample rate   as like this   test_size=0.2
and changed the default split rate I used.

I also discussed changing the default threshold of my sigmoid function for the output layer. The default is 0.5, but during my testing of data I could set it to perhaps 0.6 to stop marginal lending.

#sigmoid_output = nn.predict(X_test)
#threshold = 0.6
#predictions = (sigmoid_output >= threshold).astype(int)

If I were to model (train) and then test the model with the default value of 0.5, I could then change the threshold for assessing new data.

def custom_activation(x):
    threshold = 0.6
    return tf.keras.backend.cast(tf.keras.backend.greater_equal(x, threshold), tf.float32)

# Create the model with the custom activation function
nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))
nn.add(tf.keras.layers.Dense(units=1, activation=custom_activation))

*) Summary:   As our PCA method shows, the variance can be measured in very few pseudo columns. Reducing the columns to the necessary ones would save much of the time of a bloated neural network. But that requires a backward step to make sense of the data.
The problem with approving loans is that on a binary yes/no judgement, marginally acceptable loans can be a problem. Please see my comments just above on setting the threshold value for approval from 0.5 to 0.6. I do not recommend accepting all loans that will be predicted as to be approved.









