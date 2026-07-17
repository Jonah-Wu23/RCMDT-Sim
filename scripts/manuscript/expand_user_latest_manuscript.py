from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ADDITIONS = {
    "Rule C trades retention": ((
        "Reproducibility also depends on treating simulator failures as part of the protocol rather than hiding them "
        "inside the objective. A successful evaluation requires the declared SUMO inputs, deterministic seed, and a "
        "complete route-chain output. Missing or malformed output is retried under the same declared policy and is not "
        "replaced by a convenient numerical score. The 40-evaluation budget therefore counts successful evaluations, "
        "while failed attempts remain visible in the run log. The comparison controls extend to stochastic inputs. "
        "L1 BO and continued LHS share the initial 15 evaluations, and L2 mechanisms share priors, ensemble members, "
        "and member seeds wherever the mechanism matrix requires equivalence. Observation keys and audit thresholds "
        "are frozen before the cross-day simulation is inspected. These choices do not remove simulation uncertainty; "
        "they make its source easier to trace. A later reproduction can reconstruct which observation population was "
        "used, which stochastic realization generated each output, which attempts failed, and which successful "
        "evaluations contributed to the reported summaries."
    ), (
        "These limits also affect how the audit metrics should be read. Retention measures coverage, whereas K-S "
        "measures distributional discrepancy only on supported retained keys; neither quantity substitutes for the "
        "other. The cross-day clean sample has 29 real events, compared with 60 in development, and IRN matches are "
        "sparse for flagged keys. Holding, recovery time, and terminal behavior enter through the audit and frozen "
        "bus/stop parameters rather than a detailed control policy. Operator actions therefore remain partly "
        "identified. A larger multi-day sample would support route-specific threshold checks, more stable tail "
        "estimates, and a clearer separation between recurring operating patterns and day-specific disturbances."
    )),
    "RCMDT separates bus/stop calibration": (
        "The comparisons also indicate where the reported gains originate. A1, which changes only L1, has the "
        "smallest cross-day full-window K-S, while A2 remains close to A0. A3 and A4 use different L2 observation "
        "semantics and reach similar cross-day means of 0.351 and 0.348. Their worst-window discrepancies remain "
        "large. Likewise, BO improves the mean final L1 score and wins four paired seeds, but seed 3 favors LHS. "
        "Taken together, these results support the semantic and scope controls in the protocol without implying that "
        "each added stage or optimizer choice is uniformly superior. Future work should preserve the same frozen-key "
        "and equal-budget comparisons while adding more operating days, explicit dispatch and holding records, and "
        "route-level checks of the audit thresholds. Operationally, the calibrated parameters should be interpreted "
        "as scenario-level quantities under the declared observation contract, not as direct estimates of individual "
        "driver or controller decisions. The frozen-key audit makes this distinction inspectable: a configuration is "
        "judged against the same retained observational support, and unsupported keys remain visible instead of being "
        "absorbed into an aggregate score. This is especially important when retention and distributional agreement "
        "move in different directions. A method that keeps more keys can still have a larger K-S distance, while a "
        "lower K-S value can be obtained on narrower support. Reporting both prevents either outcome from being read "
        "as an unconditional improvement. The same discipline should apply to later extensions. New control variables "
        "or data-assimilation stages should be introduced through explicit ablations, shared stochastic inputs where "
        "comparability requires them, and validation on calendar days that do not enter calibration. This would test "
        "whether the present cross-day patterns persist without weakening the failure and coverage diagnostics that "
        "make the protocol auditable."
    ),
}


def clone_paragraph_after(paragraph, text: str) -> None:
    source = paragraph._p
    target = OxmlElement("w:p")
    p_pr = source.find(qn("w:pPr"))
    if p_pr is not None:
        target.append(deepcopy(p_pr))
    run = OxmlElement("w:r")
    if paragraph.runs and paragraph.runs[0]._r.rPr is not None:
        run.append(deepcopy(paragraph.runs[0]._r.rPr))
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    target.append(run)
    source.addnext(target)


def build(source: Path, output: Path) -> None:
    document = Document(source)
    matched: set[str] = set()
    for paragraph in list(document.paragraphs):
        normalized = " ".join(paragraph.text.split())
        for prefix, addition in ADDITIONS.items():
            if normalized.startswith(prefix):
                texts = (addition,) if isinstance(addition, str) else addition
                for text in reversed(texts):
                    clone_paragraph_after(paragraph, text)
                matched.add(prefix)
                break
    missing = set(ADDITIONS) - matched
    if missing:
        raise RuntimeError(f"Could not find insertion anchors: {sorted(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
