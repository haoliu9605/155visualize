import basic_vis
from visualization import visualize
import recsys_5A
import recsys_5B
import recsys_5B_adv
import recsys_5C
import os

U, V = recsys_5A.getUV()
os.mkdir("recsys_5A")
visualize(U, V, "recsys_5A")

U, V = recsys_5B.getUV()
os.mkdir("recsys_5B")
visualize(U, V, "recsys_5B")

U, V = recsys_5B_adv.getUV()
os.mkdir("recsys_5B_adv")
visualize(U, V, "recsys_5B_adv")

U, V = recsys_5C.getUV()
os.mkdir("recsys_5C")
visualize(U, V, "recsys_5C")
