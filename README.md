# ParetoCSP2
Github repository for our manuscript - **"Polymorphism crystal structure prediction with adaptive space group diversity control"**

Authors: Sadman Sadeed Omee, Lai Wei, Sourin Dey, and Jianjun Hu. <br>

[Machine Learning and Evolution Laboratory,](http://mleg.cse.sc.edu)<br />
Department of Computer Science and Engineering, <br />
University of South Carolina,<br/>
SC, 29201, United States.

Crystalline materials can form different structural arrangements (i.e. polymorphs) with the same chemical composition, exhibiting distinct physical properties depending on how they were synthesized or the conditions under which they operate. For example, carbon can exist as graphite (soft, conductive) or diamond (hard, insulating). Computational methods that can predict these polymorphs are vital in materials science, which help understand stability relationships, guide synthesis efforts, and discover new materials with desired properties. 

However, effective crystal structure prediction (CSP) algorithms for inorganic polymorph structures remain limited. We propose **ParetoCSP2**, a  multi-objective genetic algorithm for polymorphism CSP that incorporates an adaptive space group diversity control technique and the sustainable age-fitness Pareto optimized evolutionary mechanism, preventing over-representation of any single space group in the population guided by a neural network interatomic potential. Using an improved population initialization method and performing iterative structure relaxation, ParetoCSP2 not only alleviates premature convergence but also achieves improved convergence speed.

# Table of Contents
* [Necessary Installations](#installation)
* [How to run](#usage)
* [Contributors](#contributors)
* [Acknowledgement](#acknowledgement)

<a name="installation"></a>
## Necessary Installations
Please install the following packages if not already installed. We show how to install them using **pip** only, but you can also use **conda** for the installation purpose. Also you can a virtual environment using conda or pip for this purpose (recommended).

Use the following commands to install the necessary packages:
```bash
git clone https://github.com/usccolumbia/ParetoCSP2.git
cd ParetoCSP2
pip install -r requirements.txt
```

<a name="usage"></a>
## How to run
The default configuration are mentioned in the ```config.py``` file. So simply running the following command will run the algorithm will all default configuration (including compostion and algorithm name):
```bash
python predict_structure.py
```
For **regular CSP**, use the following command (recommended):
```bash
python predict_structure.py --comp=Sr1Ti1O3 --alg=paretocsp2 --mode=csp --pop=100 --max_step=500
```
This will run ParetoCSP2 for regular CSP with the **unit cell composition** Sr<sub>1</sub>Ti<sub>1</sub>O<sub>3</sub> (you need to specify the atom number after each compound symbol), a **population size** of 100, and for a total 500 **generations**.

For **polymorphism CSP**, use the following command (recommended):
```bash
python predict_structure.py --comp=Sr1Ti1O3 --alg=paretocsp2 --mode=poly --pop=100 --num_track=10 --num_copy=3 --max_step=500
```
This will run ParetoCSP2 for polymorphism CSP with the **unit cell composition** Sr<sub>1</sub>Ti<sub>1</sub>O<sub>3</sub> (you need to specify the atom number after each compound symbol), a **population size** of 100, and for a total 500 **generations**. The algorithm will keep track of the best 10 different space group structures (based on structures' total energy), and within each space group it will keep track of at most 3 different stuctures with that space group.

Other arguments that can be passed with the command are mentioned in ```predict_structure.py``` file. You can try different combinations of population size, number of generations, crossover probability, mutation probability, seed, etc. to get different results for both regular CSP and polymorphism CSP.

<a name="contributors"></a>
## Contributors

1. Sadman Sadeed Omee (<https://www.sadmanomee.com/>)
2. Dr. Jianjun Hu (<http://www.cse.sc.edu/~jianjunh/>)

## Acknowledgement

Our code is based on the [GN-OA](http://www.comates.group/links?software=gn_oa) algorithm's repository, which has a well-developed CSP code. We also used the [Pymoo](https://github.com/anyoptimization/pymoo) code's repository for implementing the AFPO-enhanced NSGA-III.
