"""
CEC 2017 function descriptions and landscape priors.
Extended with offline-computed landscape properties to provide
the ReAct agent with informative priors about the optimization problem.
"""
try:
    from .cec2017_py.cec2017.functions import all_functions
except ImportError:
    from cec2017_py.cec2017.functions import all_functions
# Offline landscape priors for CEC 2017 functions (pre-computed at full bounds).
# These provide the ReAct agent with an initial understanding of the problem structure.
#
# Categories from the CEC 2017 specification:
#   f1-f3:   unimodal
#   f4-f10:  simple multimodal
#   f11-f20: hybrid
#   f21-f30: composition

CEC2017_META = {
    "f1":  {"modality": "unimodal",       "separability": "non_separable",   "ruggedness": "low",    "basin_type": "broad_valley"},
    "f2":  {"modality": "unimodal",       "separability": "partially_separable", "ruggedness": "low", "basin_type": "broad_valley"},
    "f3":  {"modality": "unimodal",       "separability": "non_separable",   "ruggedness": "low",    "basin_type": "broad_valley"},
    "f4":  {"modality": "multimodal_few", "separability": "non_separable",   "ruggedness": "medium", "basin_type": "narrow_valley"},
    "f5":  {"modality": "multimodal_few", "separability": "partially_separable", "ruggedness": "medium", "basin_type": "moderate"},
    "f6":  {"modality": "multimodal_few", "separability": "non_separable",   "ruggedness": "medium", "basin_type": "narrow_valley"},
    "f7":  {"modality": "multimodal_many","separability": "non_separable",   "ruggedness": "high",   "basin_type": "narrow_valley"},
    "f8":  {"modality": "multimodal_many","separability": "non_separable",   "ruggedness": "high",   "basin_type": "narrow_valley"},
    "f9":  {"modality": "multimodal_many","separability": "non_separable",   "ruggedness": "high",   "basin_type": "narrow_valley"},
    "f10": {"modality": "multimodal_many","separability": "non_separable",   "ruggedness": "high",   "basin_type": "narrow_valley"},
    "f11": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "medium", "basin_type": "mixed"},
    "f12": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "high",   "basin_type": "mixed"},
    "f13": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "high",   "basin_type": "mixed"},
    "f14": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "medium", "basin_type": "mixed"},
    "f15": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "high",   "basin_type": "mixed"},
    "f16": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "high",   "basin_type": "mixed"},
    "f17": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "medium", "basin_type": "mixed"},
    "f18": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "high",   "basin_type": "mixed"},
    "f19": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "high",   "basin_type": "mixed"},
    "f20": {"modality": "hybrid",         "separability": "non_separable",   "ruggedness": "high",   "basin_type": "mixed"},
    "f21": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "high",   "basin_type": "complex"},
    "f22": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "high",   "basin_type": "complex"},
    "f23": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "high",   "basin_type": "complex"},
    "f24": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "high",   "basin_type": "complex"},
    "f25": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "high",   "basin_type": "complex"},
    "f26": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "high",   "basin_type": "complex"},
    "f27": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "high",   "basin_type": "complex"},
    "f28": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "high",   "basin_type": "complex"},
    "f29": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "very_high","basin_type": "complex"},
    "f30": {"modality": "composition",    "separability": "non_separable",   "ruggedness": "very_high","basin_type": "complex"},
}


class Feature:
    """
    Loads CEC 2017 function names and descriptions.
    Extended with landscape priors for function-aware ReAct prompting.
    """

    def __init__(self):
        self.all_function = all_functions.copy()
        self.descriptions = {}

    def getAllFuncName(self):
        for func in self.all_function:
            func_name = func.__name__
            doc = func.__doc__

            if doc:
                doc_lines = [line.strip() for line in doc.strip().split('\n') if line.strip()]
                describe = doc_lines[0]
            else:
                describe = "None"

            self.descriptions[func_name] = describe

    def get_landscape_prior(self, func_name: str) -> str:
        """Return a human-readable landscape prior for a CEC 2017 function.

        Used in ReAct system prompts to give the LLM prior knowledge about
        the problem structure.
        """
        meta = CEC2017_META.get(func_name)
        if meta is None:
            return (
                f"Function '{func_name}': landscape properties are unknown. "
                f"Rely on online landscape analysis during optimization."
            )

        return (
            f"Function '{func_name}': "
            f"known to be {meta['modality']}, "
            f"{meta['separability']}, "
            f"with {meta['ruggedness']} ruggedness "
            f"and {meta['basin_type']} basin structure. "
            f"Description: {self.descriptions.get(func_name, 'N/A')}"
        )

    def get_function_meta(self, func_name: str) -> dict:
        """Return the metadata dict for a function, or empty dict."""
        return CEC2017_META.get(func_name, {})
