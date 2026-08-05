## Description: <br>
Use for reusable Holoscan Module work with ./holohub: scaffold, tests, editable install, DEB/WHEEL packaging, and clean-consumer proof. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to scaffold, develop, test, package, and prove reusable Holoscan Modules through the HoloHub CLI lifecycle. <br>

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
- [HoloHub CLI Contract](references/holohub-cli-contract.md) <br>
- [Module Development](references/module-development.md) <br>
- [Module Consumer and Packaging](references/module-consumer-packaging.md) <br>
- [Create a Module Tutorial](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan-modules/create-a-module) <br>
- [Use a Module Tutorial](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan-modules/use-a-module) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (2 positive, 1 negative) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helps complete the user's goal and expected workflow. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>
- `goal_accuracy`: Verifies the user's goal was achieved. <br>
- `behavior_check`: Verifies the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 58% → 89% (+31 points) | 47% → 74% (+27 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 33% → 80% (+47 points) | 33% → 67% (+33 points) |
| Discoverability | 66% → 100% (+34 points) | 50% → 81% (+31 points) |
| Effectiveness | 34% → 67% (+32 points) | 18% → 57% (+39 points) |
| Efficiency | 58% → 100% (+42 points) | 33% → 67% (+33 points) |

## Skill Version(s): <br>
bc84ed7f (source: git SHA, committed 2026-08-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
