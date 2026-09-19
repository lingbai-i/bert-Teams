# 团队协作流程

本文档描述本仓库的分支模型、日常开发流程与发布流程。代码规范、提交规范、注释规范等代码层面的要求见 [Agent.md](./Agent.md)，本文档不再重复。

## 分支模型

| 分支 | 用途 | 保护规则 |
|------|------|----------|
| `main` | 稳定发布线，只保留经过验证、可正常运行的代码 | 禁止直接 push；必须走 PR 且经负责人审核批准后合并 |
| `dev` | 开发集成分支，所有功能开发的合流点 | 禁止直接 push；必须走 PR，无需审核即可合并 |
| 个人开发分支 | 从 `dev` 拉出，用于具体功能或修复的开发 | 无保护，成员可自由 push |

规则要点：

- `main` 与 `dev` 禁止直接提交、禁止 force push、禁止删除分支，以上均由服务端强制拦截。
- 任何代码变更必须以 PR 形式进入 `dev`，保证每一条变更有记录、有 diff 可查。
- `main` 只接受来自 `dev` 的合并，不接受其他来源。
- 个人分支命名建议：`<姓名>/<功能描述>` 或 `<类型>/<描述>`，例如 `zhangsan/login-feature`、`feature/login`、`fix/parser-crash`。

## 首次准备

1. 负责人在仓库 Settings → Collaborators 中邀请成员；成员接受邀请后获得写权限。
2. 成员克隆仓库：

```bash
git clone https://github.com/lingbai-i/bert-Teams.git
cd bert-Teams
```

## 日常开发流程

### 1. 同步 `dev` 并拉出个人分支

每次开始新任务前，先同步 `dev`，再从最新位置拉分支，避免基于过期代码开发：

```bash
git checkout dev
git pull origin dev
git checkout -b zhangsan/login-feature
```

### 2. 开发与提交

开发过程中遵守 [Agent.md](./Agent.md) 的各项规范，特别注意：

- commit message 使用简体中文，遵循 `<type>(<scope>): <description>` 格式，例如 `feat(auth): 新增登录接口`。
- 一个 commit 只包含一个逻辑完整的变更，较大的改动拆分为多个小 commit。
- 提交前确认代码可正常运行，禁止提交明显无法通过基本验证的代码。

```bash
git add <文件>
git commit -m "feat(auth): 新增登录接口"
```

### 3. 推送个人分支

```bash
git push -u origin zhangsan/login-feature
```

若向 `main` 或 `dev` 直接 push，会被服务端拒绝并提示 protected branch，这是预期行为。

### 4. 发起 PR（目标分支为 `dev`）

新建 PR 时目标分支默认即 `dev`；命令行方式如下：

```bash
gh pr create --base dev --title "feat: 登录功能" --body "功能说明"
```

### 5. 合并

`dev` 上的 PR 无需审核，创建后即可自行合并（网页上的合并按钮，或 `gh pr merge`）。合并后可删除个人分支：

```bash
gh pr merge
git push origin --delete zhangsan/login-feature
```

## 发布流程（负责人执行）

`dev` 上的代码经过验证、达到可发布状态后，发起 `dev` → `main` 的 PR：

```bash
gh pr create --base main --head dev --title "release: 合并 dev 至 main"
gh pr merge
```

该 PR 需要负责人审核批准后才能合并，因此 `main` 上始终只保留经过审核的稳定版本。

## 常见问题

- **push 被拒绝（remote rejected / protected branch）**：目标分支是受保护的 `main` 或 `dev`，请改为推送到个人分支并发起 PR。
- **PR 的目标分支选错了**：在 PR 页面底部的 Edit 处可直接修改目标分支，无需关闭重建。
- **本地 `dev` 落后远程**：先执行 `git checkout dev && git pull origin dev`，再从中拉出新分支。
- **需要同步他人已合并的改动**：在个人分支上执行 `git fetch origin && git rebase origin/dev`，解决冲突后重新 push；若该分支已有 PR，PR 会自动更新。
- **误在个人分支之外直接改了本地 `dev`**：若尚未 push，可直接丢弃本地改动（`git checkout .` 或 `git restore .`），按标准流程重新拉分支。
