import basic_vis
import visualization_5A
import visualization_5B
import visualization_5B_adv
import visualization_5C
import recsys_5A
import recsys_5B
import recsys_5B_adv
import recsys_5C
import numpy as np
import os

U, V = recsys_5A.getUV()
os.makedirs("recsys_5A",exist_ok = True)
np.save('recsys_5A/U_A',U)
np.save('recsys_5A/V_A',V)
visualization_A.visualize(U, V, "recsys_5A")

U, V = recsys_5B.getUV()
os.makedirs("recsys_5B",exist_ok = True)
np.save('recsys_5B/U_B',U)
np.save('recsys_5B/V_B',V)
visualization_5B.visualize(U, V, "recsys_5B")

U, V = recsys_5B_adv.getUV()
os.makedirs("recsys_5B_adv",exist_ok = True)
np.save('recsys_5B_adv/U_B_adv',U)
np.save('recsys_5B_adv/V_B_adv',V)
visualization_5B_adv.visualize(U, V, "recsys_5B_adv")

U, V = recsys_5C.getUV()
os.makedirs("recsys_5C",exist_ok = True)
np.save('recsys_5C/U_C',U)
np.save('recsys_5C/V_C',V)
visualization_5C.visualize(U, V, "recsys_5C")
