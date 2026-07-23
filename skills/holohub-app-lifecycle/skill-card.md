## Description: <br>

Use for non-failing HoloHub app work with ./holohub: scaffold, build, run, test, visual evidence, lint, and flow benchmarking. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner

NVIDIA <br>

### License/Terms of Use: <br>

Apache-2.0 <br>

## Use Case: <br>

Developers and engineers building, testing, and maintaining HoloHub applications through the ./holohub CLI workflow. <br>

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

- [Application Workflow](references/application-workflow.md) <br>
- [Flow Benchmarking](references/flow-benchmarking.md) <br>
- [HoloHub CLI Contract](references/holohub-cli-contract.md) <br>
- [NVIDIA HoloHub](https://github.com/nvidia-holoscan/holohub) <br>

## Skill Output: <br>

**Output Type(s):** [Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>

- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>

## Evaluation Tasks: <br>

Evaluated against 4 evaluation tasks (2 positive skill-activation, 2 negative activation) in k8s-sandbox environment. <br>

## Evaluation Metrics Used: <br>

Reported benchmark dimensions: <br>

- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>

- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>

## Evaluation Results: <br>

| Dimension | Num | Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) | Codex (`openai/openai/gpt-5.5`) |
| --- | ---: | ---: | ---: |
| Security | 4 | 100% (+0%) | 100% (+25%) |
| Correctness | 4 | 95% (+50%) | 80% (+20%) |
| Discoverability | 4 | 100% (+25%) | 73% (+12%) |
| Effectiveness | 4 | 82% (+44%) | 66% (+44%) |
| Efficiency | 4 | 100% (+34%) | 75% (+21%) |

## Skill Version(s): <br>

387c2d13 (source: git SHA, committed 2026-07-23) <br>

## Ethical Considerations: <br>

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
