## Description: <br>
Use for reusable Holoscan Module work with ./holohub: scaffold, tests, editable install, DEB/WHEEL packaging, and clean-consumer proof. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building reusable Holoscan Modules using HoloHub, covering scaffolding, testing, editable install, DEB/WHEEL packaging, and clean-consumer proof workflows. <br>

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
- [Create a Module Tutorial](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan-modules/create-a-module) <br>
- [Use a Module Tutorial](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan-modules/use-a-module) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative) in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the right skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 45% → 84% (+39 points) | 41% → 58% (+17 points) |
| Security | 50% → 100% (+50 points) | 100% → 100% (±0 points) |
| Correctness | 35% → 75% (+40 points) | 20% → 40% (+20 points) |
| Discoverability | 56% → 97% (+41 points) | 36% → 70% (+34 points) |
| Effectiveness | 24% → 54% (+30 points) | 17% → 17% (±0 points) |
| Efficiency | 58% → 95% (+36 points) | 33% → 64% (+31 points) |

## Skill Version(s): <br>
709d53ca (source: git SHA, committed 2026-08-31) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
