Installation Steps: 

1. Download the git repo via command : 
***git clone https://github.com/Metamorph-KTH/CFG_Cloning.git***
   Later go to the folder CFG_Cloning by commnand: 
   ***cd CFG_Cloning***

2. Create a python 3 virtual environment to run so that it doesnt disturb other program's dependencies. Lets name the virtual environment as clone_env
   Command: ***virtualenv -p python3 clone_env***

3. Activate the environement using the command :
***source ./clone_env/bin/activate***

4. Once the virtual environement is setup, its time to install the dependencies. It is found in requirements.txt 
   Command: ***pip3 install -r requirements.txt*** 

5. Now its time to generate the graph similarity jsons. The input graphs are provided in folder DotFiles. In case you want to add your own graph files, make sure to add the graph filenames to filenames.txt before executing. **(*Note this is very important)** You can do this by command:
***ls ./DotFiles > filenames.txt***  
   To run the graph.py file using command : 
   ***python3 graph.py***
   
6.  The jsons created will be stored jsonFiles folder. We could create a training and testing set of these json files and use it to run the SIMGNN program.   

