"""Tasks for compiling the paper and presentation(s)."""

import shutil

import pytask
from pytask_latex import compilation_steps as cs
from topics_causal_inf.config import BLD, DOCUMENTS, ROOT
from topics_causal_inf.generic_ml.task_generic_ml_plots import (
    ID_TO_KWARGS_CLANS,
    ID_TO_KWARGS_GATES,
)
from topics_causal_inf.generic_ml.task_generic_ml_tables import (
    ID_TO_KWARGS,
    ID_TO_KWARGS_COVERAGE,
)
from topics_causal_inf.wa_replication.task_wa_replication_tables import (
    ID_TO_KWARGS_TABLES,
)

documents = ["paper"]

path_genml_tables = [args.path_to_table for _, args in ID_TO_KWARGS.items()]
path_genml_tables_coverage = [
    args.path_to_table for _, args in ID_TO_KWARGS_COVERAGE.items()
]
path_genml_plots_clans = [
    args["path_to_plot"] for _, args in ID_TO_KWARGS_CLANS.items()
]
path_genml_plots_gates = [
    args["path_to_plot"] for _, args in ID_TO_KWARGS_GATES.items()
]
path_wa_tables = [args.path_to_table for _, args in ID_TO_KWARGS_TABLES.items()]

dependencies = (
    path_genml_tables
    + path_genml_tables_coverage
    + path_genml_plots_clans
    + path_genml_plots_gates
    + path_wa_tables
)

for document in documents:

    @pytask.mark.latex(
        script=DOCUMENTS / f"{document}.tex",
        document=BLD / "documents" / f"{document}.pdf",
        compilation_steps=cs.latexmk(
            options=(
                "--pdf",
                "--interaction=nonstopmode",
                "--synctex=1",
                "--cd",
                "--f",
            ),
        ),
    )
    @pytask.task(id=document, kwargs={"depends_on": dependencies})
    def task_compile_document():
        """Compile the document specified in the latex decorator."""

    kwargs = {
        "depends_on": BLD / "documents" / f"{document}.pdf",
        "produces": ROOT / f"{document}.pdf",
    }

    @pytask.task(id=document, kwargs=kwargs)
    def task_copy_to_root(depends_on, produces):
        """Copy a document to the root directory for easier retrieval."""
        shutil.copy(depends_on, produces)
