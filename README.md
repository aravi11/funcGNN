
Installation Steps: 

1. Download the git repo via command : \
***git clone https://github.com/aravi11/funcGNN.git*** \
   Later go to the folder funcGNN by command: \
   ***cd funcGNN***

2. Create a python 3 virtual environment to run so that it doesnt disturb other program's dependencies. Lets name the virtual environment as clone_env. Command: \
   ***virtualenv -p python3 clone_env*** \
  If virtual environment is not installed in the system follow the following steps first: \
  ***sudo apt-get install python3-pip*** \
  ***pip3 install virtualenv***

3. Activate the environement using the command :
***source ./clone_env/bin/activate***

4. Once the virtual environement is setup, its time to install the dependencies. It is found in requirements.txt 
   Command: ***pip3 install -r requirements.txt*** 

5. To run the training program use command : 
   ***python3 src/main.py***
   
6.  The output wil be stored in the outputFiles folder. 

=======

