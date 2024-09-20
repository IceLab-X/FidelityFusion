import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from MFBO.Discrete.CFKG import discrete_fidelity_knowledgement_gradient as DMF_KG
from MFBO.Discrete.MF_EI import expected_improvement as DMF_EI
# from MFBO.Discrete.MF_ES import entropy_search as DMF_ES
from MFBO.Discrete.MF_UCB import upper_confidence_bound as DMF_UCB