import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from MFBO.Continue.CFKG import continuous_fidelity_knowledgement_gradient as CMF_KG
# from MFBO.Discrete.MF_ES import entropy_search as DMF_ES
from MFBO.Continue.MF_UCB import upper_confidence_bound_continuous as CMF_UCB