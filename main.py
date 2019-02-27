import basic_vis
from visualization import visualize
import recsys_5A
import recsys_5B
import recsys_5B_adv
import recsys_5C
import os

U, V = recsys_5B_adv.getUV()
os.mkdir("5B_adv")
visualize(U, V, "5B_adv")
