import json
import os
import zstandard as zstd

import datasets

_CITATION="""\
@article{azerbayev2023llemma,
    title={Llemma: an open language model for mathematics},
    author={Zhangir Azerbayev and Hailey Schoelkopf and Keiran Paster and Marco Dos Santos and Stephen McAleer and Albert Q. Jiang and Jia Deng and Stella Biderman and Sean Welleck},
    eprint={xyz.xyz},
    archivePrefix={arXiv}
    year={2023}
}
"""

_DESCRIPTION = """\
A dataset of high quality mathematical text. """
_HOMEPAGE = "https://github.com/EleutherAI/math-lm"


# hacky workaround: listing out files here for download, because removing dataloader script entirely introduced another bug
_ARXIV_FILES = {
    "train": [f"arXiv_{i:03}.jsonl.zst" for i in range(100)],
    "validation": [f"arXiv_{i:03}.jsonl.zst" for i in range(100)],
    "test": [f"arXiv_{i:03}.jsonl.zst" for i in range(100)],
}
_OWM_FILES = {
    "train": [f"shard-{i:04}.jsonl.zst" for i in range(63)],
    "validation": ["val.jsonl.zst"],
    "test": ["test.jsonl.zst"],
}
_ALGSTACK_FILES = {
    "train": ["agda0000.jsonl.zst", "c0000.jsonl.zst"] 
    + [f"cpp{i:04}.jsonl.zst" for i in range(5)] 
    + [f"fortran{i:04}.jsonl.zst" for i in range(4)]
    + ["gap0000.jsonl.zst"]
    + [f"github-MATLAB-train-{i:04}.jsonl.zst" for i in range(4)] 
    + [f"github-coq-train-{i:04}.jsonl.zst" for i in range(3)]
    + ["github-isabelle-train-0000.jsonl.zst", "github-lean-train-0000.jsonl.zst"]
    + ["haskell0000.jsonl.zst", "idris0000.jsonl.zst", "isa_proofsteps.jsonl.zst"]
    + [f"julia{i:04}.jsonl.zst" for i in range(6)]
    + ["jupyter-notebook0000.jsonl.zst", "lean_proofsteps.jsonl.zst", "maple0000.jsonl.zst"]
    + [f"python{i:04}.jsonl.zst" for i in range(42)]
    + ["r0000.jsonl.zst"]
    + [f"tex{i:04}.jsonl.zst" for i in range(3)],
    "validation": [
        "agda-validation.jsonl.zst",
        "c-validation.jsonl.zst",
        "cpp-validation.jsonl.zst",
        "fortran-validation.jsonl.zst",
        "gap-validation.jsonl.zst",
        "github-MATLAB-validation-0000.jsonl.zst",
        "github-coq-validation-0000.jsonl.zst",
        "github-isabelle-validation-0000.jsonl.zst",
        "github-lean-validation-0000.jsonl.zst",
        "haskell-validation.jsonl.zst",
        "idris-validation.jsonl.zst",
        "isa_proofsteps.jsonl.zst",
        "julia-validation.jsonl.zst",
        "jupyter-notebook-validation.jsonl.zst",
        "lean_proofsteps.jsonl.zst",
        "maple-validation.jsonl.zst",
        "python-validation.jsonl.zst",
        "r-validation.jsonl.zst",
        "tex-validation.jsonl.zst",
    ],
    "test": [
        "agda-test.jsonl.zst",
        "c-test.jsonl.zst",
        "cpp-test.jsonl.zst",
        "fortran-test.jsonl.zst",
        "gap-test.jsonl.zst",
        "github-MATLAB-test-0000.jsonl.zst",
        "github-coq-test-0000.jsonl.zst",
        "github-isabelle-test-0000.jsonl.zst",
        "github-lean-test-0000.jsonl.zst",
        "haskell-test.jsonl.zst",
        "idris-test.jsonl.zst",
        "isa_proofsteps.jsonl.zst",
        "julia-test.jsonl.zst",
        "jupyter-notebook-test.jsonl.zst",
        "lean_proofsteps.jsonl.zst",
        "maple-test.jsonl.zst",
        "python-test.jsonl.zst",
        "r-test.jsonl.zst",
        "tex-test.jsonl.zst",
    ]
}

_AGDA_FILES = {
    "train": ["agda0000.jsonl.zst"],
    "validation": ["agda-validation.jsonl.zst"],
    "test": ["agda-test.jsonl.zst"],
}
_C_FILES = {
    "train": ["c0000.jsonl.zst"],
    "validation": ["c-validation.jsonl.zst"],
    "test": ["c-test.jsonl.zst"],
}
_CPP_FILES = {
    "train": [f"cpp{i:04}.jsonl.zst" for i in range(5)],
    "validation": ["cpp-validation.jsonl.zst"],
    "test": ["cpp-test.jsonl.zst"],
}
_FORTRAN_FILES = {
    "train": [f"fortran{i:04}.jsonl.zst" for i in range(4)],
    "validation": ["fortran-validation.jsonl.zst"],
    "test": ["fortran-test.jsonl.zst"],
}
_GAP_FILES = {
    "train": ["gap0000.jsonl.zst"],
    "validation": ["gap-validation.jsonl.zst"],
    "test": ["gap-test.jsonl.zst"],
}
_GITHUB_MATLAB_FILES = {
    "train": [f"github-MATLAB-train-{i:04}.jsonl.zst" for i in range(4)],
    "validation": ["github-MATLAB-validation-0000.jsonl.zst"],
    "test": ["github-MATLAB-test-0000.jsonl.zst"],
}
_GITHUB_COQ_FILES = {
    "train": [f"github-coq-train-{i:04}.jsonl.zst" for i in range(3)],
    "validation": ["github-coq-validation-0000.jsonl.zst"],
    "test": ["github-coq-test-0000.jsonl.zst"],
}
_GITHUB_ISABELLE_FILES = {
    "train": ["github-isabelle-train-0000.jsonl.zst"],
    "validation": ["github-isabelle-validation-0000.jsonl.zst"],
    "test": ["github-isabelle-test-0000.jsonl.zst"],
}
_GITHUB_LEAN_FILES = {
    "train": ["github-lean-train-0000.jsonl.zst"],
    "validation": ["github-lean-validation-0000.jsonl.zst"],
    "test": ["github-lean-test-0000.jsonl.zst"],
}
_HASKELL_FILES = {
    "train": ["haskell0000.jsonl.zst"],
    "validation": ["haskell-validation.jsonl.zst"],
    "test": ["haskell-test.jsonl.zst"],
}
_IDRIS_FILES = {
    "train": ["idris0000.jsonl.zst"],
    "validation": ["idris-validation.jsonl.zst"],
    "test": ["idris-test.jsonl.zst"],
}
_ISA_PROOFSTEPS_FILES = {
    "train": ["isa_proofsteps.jsonl.zst"],
    "validation": ["isa_proofsteps.jsonl.zst"],
    "test": ["isa_proofsteps.jsonl.zst"],
}
_JULIA_FILES = {
    "train": [f"julia{i:04}.jsonl.zst" for i in range(6)],
    "validation": ["julia-validation.jsonl.zst"],
    "test": ["julia-test.jsonl.zst"],
}
_JUPYTER_NOTEBOOK_FILES = {
    "train": ["jupyter-notebook0000.jsonl.zst"],
    "validation": ["jupyter-notebook-validation.jsonl.zst"],
    "test": ["jupyter-notebook-test.jsonl.zst"],
}
_LEAN_PROOFSTEPS_FILES = {
    "train": ["lean_proofsteps.jsonl.zst"],
    "validation": ["lean_proofsteps.jsonl.zst"],
    "test": ["lean_proofsteps.jsonl.zst"],
}
_MAPLE_FILES = {
    "train": ["maple0000.jsonl.zst"],
    "validation": ["maple-validation.jsonl.zst"],
    "test": ["maple-test.jsonl.zst"],
}
_PYTHON_FILES = {
    "train": [f"python{i:04}.jsonl.zst" for i in range(42)],
    "validation": ["python-validation.jsonl.zst"],
    "test": ["python-test.jsonl.zst"],
}
_R_FILES = {
    "train": ["r0000.jsonl.zst"],
    "validation": ["r-validation.jsonl.zst"],
    "test": ["r-test.jsonl.zst"],
}
_TEX_FILES = {
    "train": [f"tex{i:04}.jsonl.zst" for i in range(3)],
    "validation": ["tex-validation.jsonl.zst"],
    "test": ["tex-test.jsonl.zst"],
}

_FILES_MAPPING = {
    "arxiv": _ARXIV_FILES,
    "open-web-math": _OWM_FILES,
    "algebraic-stack": _ALGSTACK_FILES,
    "C++": _CPP_FILES,
    "Agda": _AGDA_FILES,
    "Fortran": _FORTRAN_FILES,
    "C": _C_FILES,
    "GAP": _GAP_FILES,
    "matlab": _GITHUB_MATLAB_FILES,
    "Coq": _GITHUB_COQ_FILES,
    "Isabelle": _GITHUB_ISABELLE_FILES,
    "Lean": _GITHUB_LEAN_FILES,
    "Haskell": _HASKELL_FILES,
    "Idris": _IDRIS_FILES,
    "Isa_proofsteps": _ISA_PROOFSTEPS_FILES,
    "Julia": _JULIA_FILES,
    "Jupyter": _JUPYTER_NOTEBOOK_FILES,
    "Lean_proofsteps": _LEAN_PROOFSTEPS_FILES,
    "Maple": _MAPLE_FILES,
    "Python": _PYTHON_FILES,
    "R": _R_FILES,
    "Tex": _TEX_FILES,
}


class ProofPile2Config(datasets.BuilderConfig):
    """BuilderConfig for RedPajama sample."""

    def __init__(self, *args, subsets, **kwargs):
        """BuilderConfig for ProofPile2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ProofPile2Config, self).__init__(**kwargs)
        self.subsets = subsets


class ProofPile2(datasets.GeneratorBasedBuilder):
    """A large dataset of mathematical text."""
    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from ProofPile2Config
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        ProofPile2Config(
            name='default',
            subsets=['arxiv', 'open-web-math', 'algebraic-stack'],
            version=VERSION,
            description="All subsets"
        ),
        ProofPile2Config(
            name='arxiv',
            subsets=["arxiv"],
            version=VERSION,
            description="ArXiv subset"
        ),
        ProofPile2Config(
            name='open-web-math',
            subsets=['open-web-math'],
            version=VERSION,
            description="OpenWebMath"
        ),
        ProofPile2Config(
            name='algebraic-stack',
            subsets=['algebraic-stack'],
            version=VERSION,
            description="Code subset"
        ),
        ProofPile2Config(
            name='Agda',
            subsets=['Agda'],
            version=VERSION,
            description="Agda subset"
        ),
        ProofPile2Config(
            name='C++',
            subsets=['C++'],
            version=VERSION,
            description="C++ subset"
        ),
        ProofPile2Config(
            name='Fortran',
            subsets=['Fortran'],
            version=VERSION,
            description="Fortran subset"
        ),
        ProofPile2Config(
            name='C',
            subsets=['C'],
            version=VERSION,
            description="C subset"
        ),
        ProofPile2Config(
            name='GAP',
            subsets=['GAP'],
            version=VERSION,
            description="GAP subset"
        ),
        ProofPile2Config(
            name='Matlab',
            subsets=['Matlab'],
            version=VERSION,
            description="MATLAB subset"
        ),
        ProofPile2Config(
            name='Coq',
            subsets=['Coq'],
            version=VERSION,
            description="Coq subset"
        ),
        ProofPile2Config(
            name='Isabelle',
            subsets=['Isabelle'],
            version=VERSION,
            description="Isabelle subset"
        ),
        ProofPile2Config(
            name='Lean',
            subsets=['Lean'],
            version=VERSION,
            description="Lean subset"
        ),
        ProofPile2Config(
            name='Haskell',
            subsets=['Haskell'],
            version=VERSION,
            description="Haskell subset"
        ),
        ProofPile2Config(
            name='Idris',
            subsets=['Idris'],
            version=VERSION,
            description="Idris subset"
        ),
        ProofPile2Config(
            name='Isa_proofsteps',
            subsets=['Isa_proofsteps'],
            version=VERSION,
            description="ISA Proofsteps subset"
        ),
        ProofPile2Config(
            name='Julia',
            subsets=['Julia'],
            version=VERSION,
            description="Julia subset"
        ),
        ProofPile2Config(
            name='Jupyter',
            subsets=['Jupyter'],
            version=VERSION,
            description="Jupyter Notebook subset"
        ),
        ProofPile2Config(
            name='Lean_proofsteps',
            subsets=['Lean_proofsteps'],
            version=VERSION,
            description="Lean Proofsteps subset"
        ),
        ProofPile2Config(
            name='Maple',
            subsets=['Maple'],
            version=VERSION,
            description="Maple subset"
        ),
        ProofPile2Config(
            name='Python',
            subsets=['Python'],
            version=VERSION,
            description="Python subset"
        ),
        ProofPile2Config(
            name='R',
            subsets=['R'],
            version=VERSION,
            description="R subset"
        ),
        ProofPile2Config(
            name='Tex',
            subsets=['Tex'],
            version=VERSION,
            description="TeX subset"
        ),
    ]


    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "meta": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [ 
            datasets.SplitGenerator(
                name=split_obj,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_files": list(map(
                        dl_manager.download_and_extract, 
                        [
                            f"https://huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/main/algebraic-stack/{split}/{x}"
                            for subset in self.config.subsets
                            for x in _FILES_MAPPING[subset][split]
                        ]
                    ))
                },
            )
            for split, split_obj in zip(
                ("train", "validation", "test"),
                (datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST)
            )
        ]

    def _generate_examples(self, data_files):
        key = 0
        for name in data_files:
            print( "###{}####".format(name))
            try:
                with zstd.open(open(name, "rb"), "rt", encoding="utf-8") as f:
                    for x in f.readlines():
                        instance = json.loads(x)
                        if instance:
                            if "meta" not in instance:
                                instance["meta"] = dict()
                            yield key, {"text": instance["text"], "meta": json.dumps(instance["meta"])}
                            key += 1
# it seems already extracted...           
            except:
                with open(name, "r", encoding="utf-8") as f:
                    for x in f.readlines():
                        instance = json.loads(x)
                        if instance:
                            if "meta" not in instance:
                                instance["meta"] = dict()
                            yield key, {"text": instance["text"], "meta": json.dumps(instance["meta"])}
                            key += 1