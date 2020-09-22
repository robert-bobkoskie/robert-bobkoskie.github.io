
1. Put the python executables (ML_RELEARN_MDP.py, Plot_Iters.py)
   in the same directory.
2. Copy the directory, reinforcement-master to your local server.


For considereration:
Run tas follows:
1. From the directrory with source code, run ML_RELEARN_MDP.py: ./ML_RELEARN_MDP.py
2. Run the code in reinforcement-master as follows:
   a. cd to the directory: reinforcement-master
      Choose 2-d or 3-d gridworlds with this boolean: True=2-d, False=3-d
      two_d = True  #2 dimensional gridworld
   b. enter 'python gridworld.py -h' to see the available options
   c. For example, 'python gridworld.py -a value -i 100 --discount 0.9 --livingReward -1.0'
      will run value iteration for 100 iterations, gamma=0.9.