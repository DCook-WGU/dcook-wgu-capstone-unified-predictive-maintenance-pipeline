# Capstone Codex Development Kit Install

Copy the contents of this folder into:

```text
D:\wgu\Capstone_Codex_Sandbox
```

Recommended PowerShell install from your extracted folder:

```powershell
Copy-Item .\* D:\wgu\Capstone_Codex_Sandbox -Recurse -Force
cd D:\wgu\Capstone_Codex_Sandbox
.\scripts\setup_codex_outputs.ps1
git status
git add AGENTS.md CODEX_STANDARDS.md CODEX_TASK_LIBRARY.md CODEX_TRANSFER_REPORT.md CODEX_REVIEW_CHECKLIST.md codex_tasks templates scripts codex_outputs
git commit -m "Add Codex development kit"
```

Then open Codex against:

```text
D:\wgu\Capstone_Codex_Sandbox
```

First Codex prompt:

```text
Read AGENTS.md, CODEX_PROJECT_CONTEXT.md, and CODEX_STANDARDS.md. Do not modify files. Confirm you understand the sandbox rules and wait for Task 001.
```
