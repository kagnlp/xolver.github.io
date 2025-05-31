# Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team

This is a preliminary implementation of the paper "Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team". More tasks and settings will be released soon. You may see some additional debate logs [here]().

[Md. Tanzib Hosain](https://www.linkedin.com/in/tanzib-mahammad-9334b1304/),
[Salman Rahman](https://salmanrahman.net/),
[Md Kishor Morol](https://www.linkedin.com/in/kishormorol/),
[Md Rizwan Parvez](https://researchportal.hbku.edu.qa/en/persons/md-rizwan-parvez),

## Running experiments

The code for running GSM, AIME, MATH and LiveCodeBench tasks may be found in the following subfolders

* ./gsm/ contains code for running GSM
* ./aime/ contains code for running AIME
* ./math/ contains code for running MATH
* ./lcb/ contains code for running LiveCodeBench results.

**GSM:**

To generate and evaluated answer for GSM problems through Xolver, cd into the gsm directory and run:
	`gsm.py`

You can download the GSM dataset [here](https://huggingface.co/datasets/openai/gsm8k)

 **AIME:**

To generate and evaluated answer for AIME problems through Xolver, cd into the aime directory and run:
	`aime.py`

You can download the AIME datasets [here](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) and [here](https://huggingface.co/datasets/yentinglin/aime_2025) 

**MATH:**

To generate and evaluated answer for MATH problems through Xolver, cd into the math directory and run:
	`math.py`

 You can download the MATH dataset [here](https://huggingface.co/datasets/di-zhang-fdu/MATH500)

 **LiveCodeBench:**

To generate and evaluated answer for LiveCodeBench problems through Xolver, cd into the lcb directory and run:
	`lcb.py`

 You can download the LiveCodeBench dataset [here](https://huggingface.co/datasets/livecodebench/code_generation)
