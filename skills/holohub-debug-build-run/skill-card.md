## Description: <br>
Use when a concrete ./holohub command fails, hangs, regresses, or returns wrong output and needs reproducible diagnosis and verification. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers debugging concrete HoloHub wrapper command failures, hangs, regressions, or incorrect output through reproducible diagnosis and minimal fix verification. <br>

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
- [HoloHub debugging workflow](references/debug-workflow.md) <br>
- [HoloHub CLI contract](references/holohub-cli-contract.md) <br>
- [Version-sensitive diagnostic priors](references/known-issues.md) <br>
- [NVIDIA HoloHub](https://github.com/nvidia-holoscan/holohub) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Code] <br>
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
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks goal completion and expected workflow adherence. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies no unsafe operations, secret leakage, or unauthorized access. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `skill_efficiency`: Verifies routing quality and productive tool use. <br>
- `accuracy`: Verifies final-answer correctness against reference. <br>
- `goal_accuracy`: Verifies user goal was achieved. <br>
- `behavior_check`: Verifies expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Codex (Baseline → Skill Uplift) |
|---|---:|
| Overall | 73% → 75% (+2 points) |
| Security | 67% → 67% (±0 points) |
| Correctness | 80% → 87% (+7 points) |
| Discoverability | 67% → 67% (±0 points) |
| Effectiveness | 84% → 88% (+4 points) |
| Efficiency | 67% → 67% (±0 points) |

## Skill Version(s): <br>
bc84ed7f (source: git SHA, committed 2026-08-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
