"""Code for "Sparse reward subsystem in large language models" (NeurIPS 2026).

Modules, in pipeline order:

    benchmarks      load problems, parse questions / reference answers, build prompts
    answer_parsing  extract and normalise final answers from responses
    grading         math_equal answer equivalence
    outputs         step-1 output files and the string-equality labels derived from them
    text_format     chat formatting of probe inputs, paragraph splitting
    models          seeding, loading the frozen LLM
    probes          the MLP probe and its checkpoints
    value_probe     value-probe data, TD training, question-only scoring, AUC
    dopamine        TD errors at paragraph boundaries, paragraph features, dopamine probe
    pruning         neuron importance, first-layer pruning, pruning-ratio grids
    plotting        paper-style figures of the two pruning curves
"""
