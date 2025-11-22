# OSWorld Evaluation Report
**Date:** 2025-11-21 16:11

## 1. Executive Summary
- **Total Tasks:** 10
- **Solved:** 0
- **Success Rate:** 0.00%
- **Avg Steps (Success):** nan

## 2. Failure Mode Analysis
Common reasons for failure in this run:

| Failure Reason | Count |
|---|---|
| Execution Error / Wrong Action | 7 |
| Agent Gave Up (FAIL action) | 3 |

## 3. Domain Breakdown
| domain             |   Total |   Solved |   Rate |
|:-------------------|--------:|---------:|-------:|
| chrome             |       2 |        0 |      0 |
| libreoffice_calc   |       2 |        0 |      0 |
| libreoffice_writer |       2 |        0 |      0 |
| os                 |       2 |        0 |      0 |
| vscode             |       2 |        0 |      0 |

## 4. Recommendations
- **Grounding:** If 'Execution Error' is high, consider checking A11y tree truncation logic.
- **Planning:** If 'Timeout' is high, the agent may be getting stuck in loops. Consider adding a 'memory' of past actions to the prompt.
