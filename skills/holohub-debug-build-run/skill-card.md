## Description: <br>
Use when a concrete ./holohub command fails, hangs, regresses, or returns wrong output and needs reproducible diagnosis and verification. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers debugging HoloHub wrapper command failures, hangs, regressions, or incorrect output through reproducible diagnosis and minimal fix verification. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Debug Workflow](references/debug-workflow.md) <br>
- [HoloHub CLI Contract](references/holohub-cli-contract.md) <br>
- [Known Issues](references/known-issues.md) <br>
- [NVIDIA HoloHub (GitHub)](https://github.com/nvidia-holoscan/holohub) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (1 positive, 2 negative) in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 73% → 88% (+15 points) | 73% → 73% (+1 points) |
| Security | 67% → 67% (±0 points) | 67% → 67% (±0 points) |
| Correctness | 80% → 93% (+13 points) | 80% → 87% (+7 points) |
| Discoverability | 67% → 100% (+33 points) | 67% → 67% (±0 points) |
| Effectiveness | 87% → 82% (-6 points) | 83% → 80% (-3 points) |
| Efficiency | 67% → 100% (+33 points) | 67% → 67% (±0 points) |

## Skill Version(s): <br>
8e0bc39c (source: git SHA, committed 2026-07-31) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
