# Synthetic Query Generation Prompt

## Task

For each skill in the provided dataset, generate **exactly 10 diverse synthetic user queries** that a real human user might type when they need this skill. These queries will be used to build a retrieval index, so they must be realistic, diverse, and semantically rich.

## Skill Format

Each skill has:
- `name`: The skill name (e.g., `jira-project-management`, `slack-message-sender`, `docker-image-builder`)
- `description`: Detailed description of what the skill does
- `source/repo`: The GitHub repository where this skill comes from
- `installs`: Number of GitHub installations (popularity indicator)

## Output Format

Return a JSON array of 10 strings for EACH skill:

```json
{
  "skill_id": "01000001-01001110--jira-project-management",
  "skill_name": "jira-project-management",
  "queries": [
    "...",
    "...",
    // exactly 10 queries
  ]
}
```

## Generation Rules

### 1. Semantic Gap Bridging (CRITICAL)

User queries are **task-oriented, colloquial, and goal-driven**. Skill descriptions are **function-oriented, technical, and action-driven**. You must bridge this gap.

**Skill Description Example:**
> "Administer Jira projects. Use when creating/archiving projects, managing components, versions, roles, permissions, or project configuration."

**BAD queries** (too close to description, too technical):
- "Administer Jira projects" (copy-paste from description)
- "Use Jira project administration tool" (function-oriented, not task-oriented)
- "Jira project configuration management" (too formal)

**GOOD queries** (task-oriented, user perspective):
- "Set up a new project in Jira for our dev team"
- "How do I archive an old Jira project nobody uses?"
- "Need to add a new team member to our Jira project"
- "I want to organize my Jira project with components"

### 2. Diversity Requirements (CRITICAL)

The 10 queries for EACH skill MUST cover ALL of the following dimensions. **DO NOT** skip any dimension. **DO NOT** repeat patterns.

| Dimension | Description | Example for "jira-project-management" |
|-----------|-------------|----------------------------------------|
| **Direct Request** | User directly states what they want | "Set up a new Jira project" |
| **Indirect Goal** | User expresses outcome, not tool | "I need to organize our team's tasks better" |
| **Problem Statement** | User describes pain point | "Our Jira project is a mess, need help cleaning it up" |
| **How-To Question** | Question format seeking instruction | "How do I create project components in Jira?" |
| **Comparison** | Compare approaches/tools | "Should I use Jira or Trello for project tracking?" |
| **Scenario/Context** | Specific real-world situation | "Just joined a team, need access to our Jira board" |
| **Casual/Colloquial** | Informal, conversational tone | "Hey, can you help me figure out Jira for my team?" |
| **Urgent/Emotional** | Time pressure or frustration | "URGENT: need to fix Jira permissions before demo" |
| **Incomplete/Vague** | User doesn't know the right terms | "Something about managing projects in that Atlassian thing" |
| **Multi-step/Task Chain** | Part of a larger workflow | "After creating the Jira project, set up sprints and add the team" |

### 3. STRICT PROHIBITIONS (DO NOT DO THESE)

The following patterns will make queries USELESS for retrieval. You MUST avoid ALL of them.

**Prohibition 1: Template Repetition**
- ❌ BAD: All queries start with "How do I...", "I want to...", "Can you..."
- ❌ BAD: Using the same sentence structure with only noun swaps
- ✅ GOOD: Mix sentence structures freely across all 10 queries

**Prohibition 2: Description Copying**
- ❌ BAD: "Administer Jira projects" (copied from description)
- ❌ BAD: "Creating and archiving Jira projects" (rephrased description)
- ✅ GOOD: "Need to set up a new project board for our sprints"

**Prohibition 3: Technical Jargon Overuse**
- ❌ BAD: "Configure Jira project roles and permissions"
- ❌ BAD: "Manage Jira project components and versions"
- ✅ GOOD: "How do I add someone to our project so they can see tasks?"

**Prohibition 4: Semantic Near-Duplicates**
- ❌ BAD: Query 1: "How to set up Jira project"
- ❌ BAD: Query 2: "How do I configure a Jira project"
- ❌ BAD: Query 3: "Steps to create a Jira project"
- ✅ GOOD: Each query must be semantically DISTINCT, not just syntactically varied

**Prohibition 5: Missing User Perspective**
- ❌ BAD: "This skill is used for Jira project management"
- ❌ BAD: "The tool helps with Jira administration"
- ✅ GOOD: Queries must be from the USER'S point of view, not describing the skill

**Prohibition 6: Overly Generic**
- ❌ BAD: "Help with Jira" (too vague, matches EVERY Jira skill)
- ❌ BAD: "Project management" (too broad)
- ✅ GOOD: Specific enough to match this skill but not others: "Archive an old Jira project"

### 4. Quality Checklist

Before returning, verify each query against ALL criteria:

- [ ] Is it something a real user would actually type? (not a robot description)
- [ ] Does it differ SEMANTICALLY from all other 9 queries? (not just word swaps)
- [ ] Does it use NATURAL language? (not technical documentation)
- [ ] Does it reflect a TASK or GOAL? (not a feature list)
- [ ] Is it specific enough to match this skill but not unrelated skills?
- [ ] Would it have LOW cosine similarity with the skill description? (this is the POINT)

### 5. Skill Popularity Awareness

Use the `installs` field to gauge how mainstream this skill is:

- **Popular skills** (installs > 100): Users likely know the tool name. Queries can reference it directly.
  - Example: "Send a Slack message to my team" (Slack is well-known)

- **Niche skills** (installs < 10): Users may NOT know the tool name. Queries should describe the TASK without naming the tool.
  - Example: "Need to send automated notifications from my app" (not "Use FooBar notifier")

## Examples by Skill Type

### Example 1: `slack-message-sender` (Popular, installs=500+)

**Skill Description:** "Send messages to Slack channels. Supports text, blocks, attachments, and thread replies."

**BAD queries (too similar to each other):**
```
1. "How do I send a message to Slack?"
2. "How to send messages in Slack"
3. "Sending Slack messages"
4. "I want to send a Slack message"
5. "Can you help me send a message on Slack?"
```

**GOOD queries (diverse, task-oriented):**
```
1. "Send a quick update to the #dev channel about the deploy"
2. "My boss wants me to notify the team when the build finishes"
3. "How do I reply to a specific thread in Slack without losing context?"
4. "Need to broadcast a company-wide announcement"
5. "Can I send a file along with my message to the design team?"
6. "Set up a bot that posts daily standup reminders"
7. "What's the easiest way to message everyone on my team at once?"
8. "Help! I need to share this screenshot with the support channel"
9. "Send a formatted message with buttons and links to the ops channel"
10. "After the CI passes, ping the QA team in Slack"
```

### Example 2: `stripe-subscription-manager` (Technical, installs=50)

**Skill Description:** "Manage Stripe subscriptions. Create, update, cancel subscriptions, handle prorations, and manage billing cycles."

**GOOD queries:**
```
1. "A customer wants to upgrade from basic to pro plan mid-month"
2. "How do I cancel someone's subscription and handle the refund?"
3. "Need to change billing from monthly to annual for a user"
4. "My SaaS app needs to handle subscription upgrades automatically"
5. "Customer emailed saying they were charged twice this month"
6. "Set up a free trial that converts to paid after 14 days"
7. "How do prorations work when someone changes plans?"
8. "User wants to pause their subscription for 2 months"
9. "Implement subscription management for my startup"
10. "Something about handling recurring payments and plan changes"
```

## Batch Processing Instructions

Process skills in batches. For each batch:
1. Read all skills in the batch
2. For each skill, generate 10 queries following ALL rules above
3. Self-check: read through all 10 queries for each skill. If any two are semantically similar, REWRITE one.
4. Return complete JSON output

## Final Reminder

**Your goal is NOT to describe the skill. Your goal is to PREDICT what a confused, busy, non-technical user would type when they need this skill.**

Think like a user, not like a developer. Think like someone who doesn't know the tool exists but has a problem to solve.
