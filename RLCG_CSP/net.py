import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import os
import importlib
import Parameters

class BipartiteGNN(K.Model):

    '''
    Initialization of the different modules and attributes
    Attributes : 
    - embedding_size : Embedding size for the intermediate layers of the neural networks
    - cons_num_features : Number of constraint features, the constraints data matrix expected has the shape (None,cons_num_features)
    - vars_num_features : Number of variable features, the variables data matrix expected has the shape (None,vars_num_features)
    - learning_rate : Optimizer learning rate
    - activation : Activation function used in the neurons
    - initializer : Weights initializer
    '''
    def __init__(self, embedding_size = 32, cons_num_features = 2, 
        vars_num_features = 9, learning_rate = 1e-3, 
        activation = K.activations.relu, initializer = K.initializers.Orthogonal):
    
        self.seed_value = Parameters.seed
        tf.random.set_seed(self.seed_value)
        super(BipartiteGNN, self).__init__()


        self.embedding_size = embedding_size
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.learning_rate = learning_rate
        self.activation = activation
        self.initializer = initializer(seed = self.seed_value)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate) 

        # constraints embedding layer
        self.cons_embedding = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # variables/columns embedding layer
        self.var_embedding = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # NN responsible for the intermediate updates
        self.join_features_NN = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer)
        ])

        # Representations updater for the constraints, called after the agregation
        self.cons_representation_NN = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),  
        ])
        # Representations updater for the variables/columns, called after the agregation
        self.vars_representation_NN = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),  
        ])

        # NN for final output, i.e., one unit logit output
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer)
        ])

        # Build of the input shapes of all the NNs
        self.build()

        # Order set for loading/saving the model
        self.variables_topological_order = [v.name for v in self.variables]


        self.seed_value = Parameters.seed
        tf.random.set_seed(self.seed_value)
    '''
    Build function, sets the input shapes. Called during initialization
    '''
    def build(self):
        self.cons_embedding.build([None, self.cons_num_features])
        self.var_embedding.build([None, self.vars_num_features])
        self.join_features_NN.build([None, self.embedding_size*2])
        self.cons_representation_NN.build([None, self.embedding_size*2])
        self.vars_representation_NN.build([None, self.embedding_size*2])
        self.output_module.build([None, self.embedding_size])
        self.built = True

    '''
    Main function taking as an input a tuple containing the three matrices :
    - cons_features : Matrix of constraints features, shape : (None, cons_num_features)
    - edge_indices : Edge indices linking constraints<->variables, shape : (2, None)
    - vars_features : Matrix of variables features, shape : (None, vars_num_features)

    Output : logit vector for the variables nodes, shape (None,1)
    '''
    def call(self, inputs):
        # print("The code is running", inputs[0].shape)

        cons_features, edge_indices, vars_features = inputs
        # print("#######################")
        # print(cons_features)
        # print(edge_indices)
        # print(vars_features)
        # print("#######################")
        # Nodes embedding, constraints and variables
        cons_features = self.cons_embedding(cons_features)
        vars_features = self.var_embedding(vars_features)

        # ==== First Pass : Variables -> Constraints ====
        # compute joint representations
        joint_features = self.join_features_NN(
                tf.concat([
                    tf.gather(
                        cons_features,
                        axis=0,
                        indices=edge_indices[0])
                    ,
                    tf.gather(
                        vars_features,
                        axis=0,
                        indices=edge_indices[1])
                    ### change this number to edge weights (patterns)
                ],1)
        )

        # Aggregation step
        output_cons = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[0], axis=1),
            shape=[cons_features.shape[0], self.embedding_size]
        )
        # Constraints representations update
        output_cons = self.cons_representation_NN(tf.concat([output_cons,cons_features],1))



        # ==== Second Pass : Constraints -> Variables ====
        # compute joint representations
        joint_features = self.join_features_NN(
                tf.concat([
                    tf.gather(
                        output_cons,
                        axis=0,
                        indices=edge_indices[0])
                    ,
                    tf.gather(
                        vars_features,
                        axis=0,
                        indices=edge_indices[1])
                ],1)
        )

        # Aggregation step
        output_vars = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[1], axis=1),
            shape=[vars_features.shape[0], self.embedding_size]
        )
        # Variables representations update
        output_vars = self.vars_representation_NN(tf.concat([output_vars,vars_features],1))

        # ==== Final output from the variables representations (constraint nodes are ignored)
        output = self.output_module(output_vars)
        return output

    '''
    Save model and current weights to a given path
    '''
    def save_state(self, path):
        import pickle
        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)

    '''
    Load an existing model from a given path
    '''
    def restore_state(self, path):
        import pickle
        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))

    def get_config(self):
        config = {}
        for i in range(len(self.variables_topological_order)):
            config[self.variables_topological_order[i]] = self.variables[i]
        return config

    @staticmethod
    def from_config(model):
        for v_name in self.variables_topological_order:
            v = [v for v in model.variables if v.name == v_name][0]
            v.assign(config[v])
        return model
       

    '''
    Training/Test function
    Input: 
    - data : a batch of data, type : tf.data.Dataset
    - train: boolean, True if function called for training (i.e., compute gradients and update weights),
                False if called for test
    Output:
    tuple(Loss, Accuracy, Recall, TNR) : Metrics
    '''
    def train_or_test(self, data, labels, totals_0, actions_0,action_info, train=False):
        mean_loss = 0
  
        batches_counter = 0

        ###########################################################
        ### how does this data(a batch) relates to transition data?
        ###########################################################
        for batch in data:
            cons_features, edge_indices, vars_features = batch
            input_tuple = (cons_features, edge_indices, vars_features)

            total_0 = totals_0[batches_counter]

            label = labels[batches_counter]

            # action = actions_0[batches_counter].tolist()


            # all_actions = action_info[batches_counter][1].tolist()

            # print(all_actions)
            # act_index = all_actions.index(action) ## used for getting the correct loss -> only conunts the loss of the selected actions

            # When called train=True, compute gradient and update weights
            if train:
                with tf.GradientTape() as tape:
                    # Get logits from the bipartite GNN model

                    ########
                    ######### may need to change to self.call?
                    logits = self.call(input_tuple)
                    # print(total_0)
                    label[0:-total_0[0]] =  logits[0:-total_0[0]] ## do not count the loss from the nodes already in the basis
                    # print(abel[act_index])
                    # print()
                    # print(logits[-total_0[0]:-1])
                    # print()
                    loss = tf.keras.metrics.mean_squared_error(label,logits) ## should not be mean_squared_error as it's then scaled down by number of nodes
                    loss = (loss * label.shape[0]) / total_0[0]  ## this is a quick fix, as there are far less action nodes compared to 
                    ## again, do we calculate the loss using the nodes we are not selecting?

                    # print(loss)
                # Compute gradient and update weights
                grads = tape.gradient(target=loss, sources=self.variables)
                self.optimizer.apply_gradients(zip(grads, self.variables))
            # If no optimizer instance set, no training is performed, give outputs and metrics only
            else:
                logits = self.call(input_tuple)
                loss = tf.keras.metrics.mean_squared_error(label,logits)

            ## these are for classification
            # prediction = tf.round(tf.nn.sigmoid(logits))
            # correct_pred = tf.equal(prediction, label)
            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            loss = tf.reduce_mean(loss)

            # Batch loss, accuracy, confusion matrix
            mean_loss += loss
            batches_counter += 1 
            # confusion_mat += confusion_matrix(labels, prediction)

        # Batch average loss
        mean_loss /= batches_counter
        return mean_loss


