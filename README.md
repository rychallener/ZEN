# zen

When we get vanilla PLD working, we will then have branches for each testing project.

We need an event unpacker for POET.

Are we using MCcubed?

What are the uncertanties for the Markov chain?

What exists now:

  zen_funcs:
  
    zen_init: set up for PLD by finding phat and dP
    
    eclipse:  model for the eclipse portion of the light curve given orbital parameters and time stamps
    
    zen:      the function that gets fed into mc3 
    
  zen:
  
    Where the magic happens, eventually. 
    
We need:

  -call mc3 (https://github.com/pcubillos/MCcubed)

  -a config file containing initial parameters for the markov chain

  -something that reads in data and parameters, perhaps from the POET event?
