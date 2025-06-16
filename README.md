## Xolver

This website showcases our work on Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team. The website is adapted from [Nerfies website](https://nerfies.github.io).

### Project Overview
Xolver is a training-free, multi-agent reasoning framework that equips black-box LLMs with persistent, evolving memory of holistic experience. Inspired by how expert problem solvers like Olympiad teams leverage diverse experiences, Xolver integrates external retrieval, tool use, agent collaboration, self-evaluation, and iterative refinement to achieve expert-level reasoning across mathematics and programming tasks.

### Key Features
- **Multi-Agent Collaboration**: Dynamic team of specialized agents (mathematicians, programmers, verifiers) that work together
- **Holistic Experience Learning**: Dual-memory architecture combining episodic long-term memory with evolving shared memory
- **Tool-Augmented Reasoning**: Seamless integration with external tools (Python execution, code debugging)
- **Iterative Refinement**: Judge-mediated feedback and continuous improvement across iterations
- **Cross-Problem Learning**: Accumulates knowledge from solved problems to enhance future performance
- **State-of-the-Art Results**: 98.1% on GSM8K, 94.4% on AIME'24, 93.7% on AIME'25, 99.8% on Math-500, 91.6% on LiveCodeBench

### Performance Highlights
Even with lightweight backbones (QWQ-32B), Xolver often surpasses the most advanced models including:
- Qwen3-235B, Gemini 2.5 Pro, o1, o3, and o4-mini-high
- Specialized reasoning agents like OctoTools, CheatSheet, Search-o1
- With stronger backbones (o3-mini-high), achieves new best results across all benchmarks

### Resources
- [Paper](https://github.com/kraritt/Xolver/blob/main/Xolver.pdf)
- [Code](https://github.com/kagnlp/Xolver)

### Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
