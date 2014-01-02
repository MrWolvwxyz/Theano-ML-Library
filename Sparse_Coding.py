import os
import re
import theano
import string
import time

import theano.tensor as T
from theano.sandbox.linalg.ops import MatrixInverse as matrix_inverse
import numpy as np
import numpy.random as random

class sparsity_coefficient:
    #This class represents a single coefficient in the sparse representation
    
    def __init__( self, num_coefficients = 10, initial_values = None, sparsity_penalty = 0, beta = None,
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
        if initial_values == None:
            print 'Initializing to random sparsity coefficients'
            self.value = np.zeros( num_coefficients )
        else:
            print 'Using user provided inital values'
            self.value = initial_values
        
        if beta == None:
            print 'Initializing beta to 1'
            self.beta = 1
        else:
            print 'Using user proved beta value'
            self.beta = beta
        
        if epsilon == None:
            print 'Initializing epsilon to 0'
            self.epsilon = 0
        else:
            print 'Using user proved epsilon value'
            self.epsilon = epsilon
        
        #These lines compute the sparsity penalty
        if sparsity_penalty == 0:
            print 'Using L1 regularization'
            self.sparsity_penalty = np.linalg.norm( self.value, ord = 1 )
        elif sparsity_penalty == 1:
            print 'Using epsilon L1'
            self.sparsity_penalty = ( ( self.value ** 2 + self.epsilon ) ** 0.5 )
        else:
            print 'Using log regularization'
            self.sparsity_penalty = np.log( 1 + self.value ** 2 )
        
        #This computes the prior probability of the sparsity coefficient     
        self.prior = T.exp( - self.beta * self.sparsity_penalty )
        

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
            print 'Using randomly initialized base vectors'
            init_value = random.rand( vector_size )
            init_value = init_value - np.mean( init_value )
            self.value = init_value
        elif len( initial_values ) != vector_size:
            raise ValueError( 'initial values not the same dimmension as vector_size' )
        else:
            print 'Using user provided base vectors'
            self.value = initial_values

class sparsity_model:
    #This class holds all the sparsity coefficients and bases
    def __init__( self, input, num_bases = 10, sparsity_function = 0, beta_top = None,
                  epsilon_top = None, learning_rate = 0.1, initial_coefficients = None,
                  initial_bases = None ):
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
        self.learning_rate = learning_rate
        
        if initial_coefficients == None:
            #Initialize the sparsity coefficients
            coefficients = np.asarray( [ sparsity_coefficient( num_coefficients = num_bases, 
                                    sparsity_penalty = sparsity_function, beta = beta_top,
                                    epsilon = epsilon_top).value for i in range( self.num_examples ) ] )
            self.coefficients = theano.shared( value = coefficients, name = 'coefficients', borrow = True ) 
        else:
           #Initialize the sparsity coefficients
            coefficients = np.asarray( [ sparsity_coefficient( num_coefficients = num_bases, initial_values = initial_coefficients,
                                    sparsity_penalty = sparsity_function, beta = beta_top,
                                    epsilon = epsilon_top).value for i in range( self.num_examples ) ] )
            self.coefficients = theano.shared( value = coefficients, name = 'coefficients', borrow = True ) 
        
        if initial_bases == None:
            #Initialize the dictionary bases
            bases = np.asarray( [ base_vector( vector_size = self.vector_dim ).value for i in range( num_bases ) ] )
            self.bases = theano.shared( value = bases, name = 'bases', borrow = True )
        else:
            #Initialize the dictionary bases
            bases = np.asarray( [ base_vector( vector_size = self.vector_dim, initial_values = initial_bases ).value for i in range( num_bases ) ] )
            self.bases = theano.shared( value = bases, name = 'bases', borrow = True )
        
        self.variance = T.var( self.bases )
        if beta_top == None:
            self.gamma = 2 * self.variance
        else:
            self.gamma = 2 * self.variance * beta_top
        
        self.params = [ self.coefficients, self.bases ]
        
    
    def feature_sign_search( self ):
        '''
        This function runs the feature_sign_search on the coefficients while
        holding the bases clamped.
        '''
        #Declare effective zero for usefullness
        effective_zero = 1e-19
        opt_cond = np.inf
        '''
        theta[ i ] is:
            -1 if self.coefficients[ i ] < 0
            1 if self.coefficients[ i ] > 0
            0 if self.coefficients[ i ] == 0
        '''
        theta = T.sgn( self.coefficients )
        active_set = []
        #This corresponds to the gram matrix by dotting the basis vectors by it's transpose
        gram_matrix = T.dot( self.bases.T, self.bases )
        target_correlation = T.dot( self.bases.T, self.x )
        
        cost = -T.sum( ( target_correlation - T.dot( gram_matrix, self.coefficients ) ) ** 2 )
        cost_grad = T.grad( cost, self.coefficients )
        
        candidate = T.argmax( cost_grad )
        if T.gt( cost_grad[ candidate ], self.gamma ):
            print 'Found candidate  greater than gamma'
            updated_theta = T.set_subtensor( theta[ candidate ], -1 )
            active_set = active_set + candidate
        if T.lt( cost_grad[ candidate ], ( -1 * self.gamma ) ):
            print 'Found candidate less than negative gamma'
            updated_theta = T.set_subtensor( theta[ candidate ], 1 )
            active_set = active_set + candidate
        
        active_bases = theano.function( [ self.bases ], [ self.bases[ active_set[ i ] ]
                                                          for i in range( len( active_set ) ) ] )
                                                          
        active_coefficients = theano.function( [ self.coefficients ], [ self.coefficients[ active_set[ i ] ]
                                                          for i in range( len( active_set ) ) ] )
                                                          
        active_theta = [ updated_theta[ active_set[ i ] ] for i in range( len( active_set ) ) ]
        
        new_coefficients = ( matrix_inverse( T.dot( active_bases.T, active_bases ) ) *
                                    ( T.dot( active_bases.T, target_correlation )
                                    - 0.5 * self.gamma * active_theta ) )
        sign_changes = 0




