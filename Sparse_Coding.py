import os
import re
import theano
import string
import time

import theano.tensor as T
import numpy as np
import numpy.random as random

class sparsity_coefficient:
    #This class represents a single coefficient in the sparse representation
    
    def __init__( self, num_coefficients = 10, initial_value = None, sparsity_penalty = 0, beta = None,
                  epsilon = None ):
        '''
        This initialization takes:
            -an initial value which is initialized to a 0 mean gaussian
             distribution if not provided
            -a sparsity penalty function
                where:
                    0 -> L1 penalty
                    1 -> epsilon L1 penalty
                    2 -> log penalty
            -an initial beta value which is initialized to 1 if not provided
            -an optional epsilon argument for the epsilon L1 penalty which is
             initialized to 0 if not provided
        '''
        if initial_value == None:
            init_value = random.normal( size = num_coefficients )
            self.value = theano.shared( value = init_value, name = 'value',
                                        dtype = theano.config.floatX )
        
        if beta == None:
            self.beta = 1
        
        if epsilon == None:
            self.epsilon = 0
        
        #These lines compute the sparsity penalty
        if sparsity_penalty == 0:
            self.sparsity_penalty = np.linalg.norm( self.value, 1 )
        elif sparsity_penalty == 1:
            self.sparsity_penalty = ( self.value ** 2 + self.epsilon ) ** 0.5
        else:
            self.sparsity_penalty = np.log( 1 + self.value ** 2 )
        
        #This computes the prior probability of the sparsity coefficient     
        prior = np.exp( - self.beta * self.sparsity_penalty )
        self.prior = theano.shared( value = prior, name = 'prior', 
                                    dtype = theano.config.floatX)
        

class base_vector:
    #This class represents a single base vector
    def __init__( self, vector_size, initial_values = None ):
        '''    
        This initializes a single base vector and takes:
            -a vector of initial values which is initialized to a gaussian normal
             distribution if not provided
            -the size of the base vector which is a required input
        '''
        if initial_values == None:
            init_value = random.normal( size = vector_size )
            self.value = theano.shared( value = init_value, name = 'value',
                                        dtype = theano.config.floatX )
        elif len( initial_values ) != vector_size:
            raise ValueError( 'initial values not the same dimmension as vector_size' )
        else:
            init_value = initial_values
            self.value = theano.shared( value = init_value, name = 'value',
                                        dtype = theano.config.floatX )

class sparsity_model:
    #This class holds all the sparsity coefficients and bases
    def __init__( self, input, num_bases = 10, sparsity_function = 0, beta_top = None,
                  epsilon_top = None ):
        '''
        This initializes the sparsity model and takes:
            -a required input which corresponds to the ( n x k ) training data
            -a sparsity penalty function
                where:
                    0 -> L1 penalty
                    1 -> epsilon L1 penalty
                    2 -> log penalty
            -an initial beta value which is initialized to 1 if not provided
            -an optional epsilon argument for the epsilon L1 penalty which is
             initialized to 0 if not provided
        '''
        self.x = input
        self.num_examples = len( input )
        self.vector_dim = len( input[ 0 ] )
        
        #Initialize the sparsity coefficients
        coefficients = [ sparsity_coefficient( num_coefficiens = num_bases, initial_value = input[ i ],
                                sparsity_penalty = sparsity_function, beta = beta_top,
                                epsilon = epsilon_top) for i in range( self.num_examples ) ]
        self.coefficients = theano.shared( value = coefficients, name = 'coefficients',
                                        dtype = theano.config.floatX )
        
        #Initialize the dictionary bases
        bases = [ base_vector( vector_size = self.vector_dim ) for i in range( num_bases ) ]
        self.bases = theano.shared( value = bases, name = 'bases', dtype = theano.config.floatX )
        
    def feature_sign_search( self ):
        gram_matrix = T.dot( self.bases.T, self.bases )
        target = 





