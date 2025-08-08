
---

### 📄 Updated `requirements.md`  
⬇️ Copy and paste this into your current **requirements.md**

[Click to expand or collapse full updated content]

```markdown
### Requirement 13: Goose Integration

**User Story:** As a developer, I want the system to deeply integrate with Goose so that I can leverage intelligent refactoring, documentation, and testing via embedded memory and AST-aware tooling.

#### Acceptance Criteria

1. WHEN the Code Agent executes a task THEN it SHALL call Goose via Python API with proper strategy (e.g., `refactor`, `extract-method`, `apply-patch`)
2. WHEN refactoring or writing tests THEN the system SHALL use Goose memory to retrieve similar historical structures and reuse patterns
3. WHEN documentation is updated THEN the Docs Agent SHALL use Goose tools like `write-docs` or `summarize`
4. WHEN errors occur in test or build THEN QA Agent SHALL invoke Goose `fix` with relevant context to automatically retry or patch the issue
5. WHEN IDE requests are made THEN the system SHALL expose Goose capabilities via the MCP server (e.g., refactor_code, generate_tests)
6. WHEN Goose memory fails or returns no match THEN the system SHALL fallback to vector search and log confidence score mismatch
7. WHEN Goose plugins are registered THEN the system SHALL execute pre/post-processing logic and log plugin outputs for traceability

### Requirement 14: AST Memory and Code Search

**User Story:** As a developer, I want the system to use AST-aware search and semantic memory so that it can retrieve accurate and structurally relevant code for reasoning and reuse.

#### Acceptance Criteria

1. WHEN the Planner Agent breaks down tasks THEN it SHALL retrieve examples using Goose memory’s `lookup()` API
2. WHEN the Code Agent searches for context THEN it SHALL prefer AST-based memory matches over raw embeddings
3. WHEN an unknown pattern is found THEN the system SHALL check Goose memory for nearest semantic match and copy structure if confidence > threshold
4. WHEN multiple functions match a query THEN the system SHALL select based on structural similarity and local token overlap
5. WHEN a Goose memory operation fails THEN it SHALL retry with modified prompt or fallback to ChromaDB

