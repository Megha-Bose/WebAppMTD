# Moving Target Defense under Uncertainty

There are two folders: Data_Gen creates the text files. The data is obtained from Sengupta's github library. This is where you look if you want to know how the text files are created

The second folder and arguably the more important one is Expts and there are three important files here; the rest of them either store output, input or are just old files.

The three files are attacker_eps.py, attacker_BR.py and attacker_fplue.py

These three files correspond to three different attacker strategies: eps corresponds to epsilon optimal -- given some epsilon, the attacker randomly chooses between all the strategies that achieve a (1-epsilon) \times optimal utility for that round. This type assumes knowledge about what move the defender makes. 

BR corresponds to Best Response (exactly the same as that of the FPL-UE paper)

and fplue corresponds to an attacker whose strategy is decided by the fplue algorithm.

The rest of the code in these three files is the same: they correspond to six defender strategies, there are two from my write-up and one "RobustRL" which you may not understand yet but that's okay.

Good luck!

