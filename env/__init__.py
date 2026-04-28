from .mud import MUDEnv
from .ddxplus import DDXPLUSEnv
from .mimic_iv_tlp import MIMIC_IV_TLPEnv
from .mimic_iv_msr import MIMIC_IV_MSREnv
from .bird import BIRDEnv
from .spider import SPIDEREnv
from .banking77 import Banking77Env
from .sentifin import SEntFiNEnv
from .mimic_iv_mr import MIMIC_IV_MREnv
from .cmdl import CMDLEnv
from .rca import RCAEnv
from .lfd import LFDEnv

from .alfworld import ALFWorldEnv
from .alfworld_react import ALFWorldReActEnv
from .scienceworld import SeqScienceWorldEnv
from .scienceworld_react import SeqScienceWorldReActEnv
from .ehr import EHREnv
from .twowiki import TwoWikiEnv

ENV_DICT = {
    "mud": MUDEnv,
    "ddxplus": DDXPLUSEnv,
    "bird": BIRDEnv,
    "spider": SPIDEREnv,
    "banking77": Banking77Env,
    "sentifin": SEntFiNEnv,
    "mimic-iv-mr": MIMIC_IV_MREnv,
    "cmdl": CMDLEnv,
    "rca": RCAEnv,
    "lfd": LFDEnv,
    "mimic-iv-msr": MIMIC_IV_MSREnv,
    "mimic-iv-tlp": MIMIC_IV_TLPEnv,
    
    "alfworld": ALFWorldEnv,
    "alfworld-react": ALFWorldReActEnv,
    "scienceworld": SeqScienceWorldEnv,
    "scienceworld-react": SeqScienceWorldReActEnv,
    "ehr": EHREnv,
    "twowiki": TwoWikiEnv
}