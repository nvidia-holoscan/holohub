## Description: <br>
Use for reusable Holoscan Module work with ./holohub: scaffold, tests, editable install, DEB/WHEEL packaging, and clean-consumer proof. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building reusable Holoscan Modules use this skill to scaffold, test, package (DEB/WHEEL), and prove clean-consumer installation through the HoloHub CLI workflow. <br>

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
- [NVIDIA HoloHub Repository](https://github.com/nvidia-holoscan/holohub) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Code, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (1 positive, 1 negative) in isolated k8s-sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Checks final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 54% → 94% (+39 points) | 58% → 53% (-5 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 0% → 100% (+100 points) | 30% → 30% (±0 points) |
| Discoverability | 75% → 100% (+25 points) | 75% → 50% (-25 points) |
| Effectiveness | 29% → 69% (+40 points) | 36% → 35% (-1 points) |
| Efficiency | 68% → 100% (+32 points) | 50% → 50% (±0 points) |

## Skill Version(s): <br>
8e0bc39c (source: git SHA, committed 2026-07-31) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
