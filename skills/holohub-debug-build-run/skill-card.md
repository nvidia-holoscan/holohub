## Description: <br>
Use when a concrete ./holohub command fails, hangs, regresses, or returns wrong output and needs reproducible diagnosis and verification. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to diagnose and fix failing, hanging, or regressing ./holohub CLI commands with reproducible proof. <br>

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
- [HoloHub CLI contract](references/holohub-cli-contract.md) <br>
- [Debug workflow](references/debug-workflow.md) <br>
- [Version-sensitive diagnostic priors](references/known-issues.md) <br>
- [HoloHub repository](https://github.com/nvidia-holoscan/holohub) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Code] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 tasks (2 positive, 2 negative); each attempt ran in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use. <br>
- Correctness: Checks whether the answer is correct. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Checks whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 61% → 89% (+28 points) | 66% → 73% (+7 points) |
| Security | 100% → 100% (±0 points) | 100% → 75% (-25 points) |
| Correctness | 35% → 85% (+50 points) | 50% → 85% (+35 points) |
| Discoverability | 61% → 98% (+38 points) | 61% → 67% (+6 points) |
| Effectiveness | 49% → 67% (+18 points) | 58% → 65% (+7 points) |
| Efficiency | 61% → 96% (+35 points) | 59% → 70% (+11 points) |

## Skill Version(s): <br>
adf8a7ae (source: git SHA, committed 2026-09-02) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
