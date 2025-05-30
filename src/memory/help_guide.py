"""Memory system help and usage guide."""

from typing import Dict, Optional, Any

HELP_MESSAGES = {
    "overview": """
Spark Memory - LRMM (LangGraph + Redis + MCP + Memory) Next-Generation AI Memory System

🔧 Available Tools:
- m_memory: Memory save/get/search/update/delete/consolidate/lifecycle/relationship analysis
- m_state: State management and checkpoints
- m_admin: System administration and security settings
- m_assistant: Natural language command processing

🧠 LRMM Advanced Features:
- Automatic conversation-document linking (CrossMemoryBridge)
- Intelligent cross-memory search
- Memory graph analysis (MemoryAnalytics)
- AI-powered connection suggestion system
- Individual memory insights analysis

For detailed help: m_memory("help", ["action_name"]) or m_memory("help", ["feature_name"])
""",
    
    "m_memory": """
m_memory - Unified Memory Management Tool

🔵 Basic Actions:
- save: Save new memory
- get: Retrieve memory
- search: Search memories
- update: Update memory
- delete: Delete memory
- consolidate: Consolidate memories
- lifecycle: Lifecycle management

🧠 LRMM Advanced Actions:
- link_conversation: Automatic conversation-document linking
- find_cross_memory: Cross-memory search
- analyze_memory_graph: Memory graph analysis
- suggest_connections: AI connection suggestions
- get_memory_insights: Individual memory insights

🆘 Help:
- help: Display help

Examples:
# Basic functions
m_memory("save", ["projects", "ai"], "new idea")
m_memory("search", [], "Redis", {"type": "keyword"})

# LRMM advanced features  
m_memory("link_conversation", ["stream:memory:conversation:2024/01/15/10/30/00"])
m_memory("find_cross_memory", ["json:memory:document:2024/01/15/report.pdf"], {"advanced": True})
m_memory("analyze_memory_graph", [])
m_memory("suggest_connections", [])
""",
    
    "search": """
Search Action Detailed Guide

Search Types:
1. keyword (Keyword Search)
   - Default search method
   - Enter search term in content
   
2. time_range (Time Range Search)
   - Specify time conditions in filters
   - date: Specific date (YYYY-MM-DD)
   - from/to: Time range (ISO datetime)
   - start_time/end_time: Alternative format

Examples:
# Keyword search
m_memory("search", [], "project")

# Today's memories search
m_memory("search", [], None, {
    "type": "time_range",
    "filters": {"date": "2025-05-28"}
})

# Time range search
m_memory("search", [], None, {
    "type": "time_range", 
    "filters": {
        "from": "2025-05-01T00:00:00",
        "to": "2025-05-31T23:59:59"
    }
})
""",
    
    "m_consolidate": """
m_consolidate - Memory Consolidation Tool

Consolidation Types:
1. path: Path-based consolidation
   - Merge similar memories in specified path
   
2. duplicate: Duplicate detection
   - Remove completely identical memories
   
3. temporal: Time-based consolidation
   - Group memories by time units

Examples:
# Path-based consolidation
m_consolidate("path", ["2025", "05"])

# Duplicate removal
m_consolidate("duplicate")

# Time-based consolidation
m_consolidate("temporal", [], None, {
    "time_buckets": ["1d", "7d", "30d"]
})
""",
    
    "m_lifecycle": """
m_lifecycle - Lifecycle Management Tool

Actions:
1. evaluate: Importance evaluation
   - Calculate memory importance score
   - Auto-adjust TTL
   
2. archive: Archive old memories
   - Store memories older than threshold_days
   
3. restore: Restore from archive
   - Activate archived memories
   
4. stats: View statistics
   - Overall memory status

Examples:
# Importance evaluation
m_lifecycle("evaluate", ["projects", "ai", "idea1"])

# Archive memories older than 90 days
m_lifecycle("archive", options={"threshold_days": 90})

# View statistics
m_lifecycle("stats")
""",
    
    "m_assistant": """
m_assistant - Natural Language Command Processing

Supported Commands:
- Save: "save this", "record this"
- Search: "find this", "what was that?"
- Summary: "summarize this", "organize this"
- Analysis: "analyze memory patterns"

Time Expressions:
- today, yesterday, this week, last week, this month
- Specific dates (2025-05-28)

Category Auto-classification:
- work: business, projects, meetings
- personal: personal, daily life
- study: learning, education
- idea: ideas, plans
- reference: references, links

Examples:
m_assistant("save today's meeting content")
m_assistant("find what I did yesterday")
m_assistant("summarize this week's work")
""",
    
    "consolidate": """
consolidate Action - Memory Consolidation (within m_memory)

Consolidation Types:
1. path: Path-based consolidation
   - Merge similar memories in specific path
   
2. duplicate: Duplicate detection
   - Find duplicates based on SHA256 hash
   
3. temporal: Time-based consolidation
   - Group and consolidate by time units

Examples:
# Path-based consolidation
m_memory("consolidate", ["2024", "01"], None, {"type": "path"})

# Duplicate detection
m_memory("consolidate", [], "content that might be duplicated", {"type": "duplicate"})

# Time-based consolidation (1 hour, 6 hours, 1 day units)
m_memory("consolidate", ["notes"], None, {
    "type": "temporal",
    "time_buckets": ["1h", "6h", "1d"]
})
""",
    
    "lifecycle": """
lifecycle Action - Lifecycle Management (within m_memory)

Tasks:
1. evaluate: Importance evaluation
   - Calculate memory importance score
   - Auto-adjust TTL
   
2. archive: Archive old memories
   - Store memories older than threshold_days
   
3. restore: Restore from archive
   - Activate archived memories
   
4. stats: View statistics
   - Overall memory status

Examples:
# Importance evaluation
m_memory("lifecycle", ["projects", "ai", "idea1"], None, {"action": "evaluate"})

# Set user importance (0.0-1.0)
m_memory("lifecycle", ["important", "doc"], 0.9, {"action": "evaluate"})

# View statistics
m_memory("lifecycle", [], None, {"action": "stats"})
""",
    
    "m_admin": """
m_admin - System Administration and Security

System Actions:
- status: Check status
- config: View configuration
- backup: Create backup
- clean: Clean data

Security Actions (security_ prefix):
- security_create_principal: Create principal
- security_grant: Grant permissions
- security_revoke: Revoke permissions
- security_api_key: API key management
- security_audit: Audit logs
- security_report: Security reports

Examples:
# System status
m_admin("status")

# Create principal
m_admin("security_create_principal", [], {"id": "user1", "roles": ["user"]})

# Grant permissions
m_admin("security_grant", ["projects", "ai"], {
    "principal_id": "user1",
    "permissions": ["read", "write"]
})
""",

    # 🧠 LRMM Advanced Features Help
    "link_conversation": """
🔗 link_conversation - Automatic Conversation-Document Linking

Features:
- Automatically detect document references in conversations
- Extract explicit/implicit/entity-based references
- Create bidirectional connection relationships and calculate strength
- Support multilingual reference patterns

Usage:
m_memory("link_conversation", [conversation_id])

Examples:
# Automatically link documents referenced in specific conversation
m_memory("link_conversation", ["stream:memory:conversation:2024/01/15/10/30/00"])

Return Value:
List of linked document IDs and connection strength

Auto-detection Patterns:
- File names: "performance_analysis.pdf", "redis_guide.md"
- Document titles: "Redis Performance Analysis Report"
- Reference expressions: "previously mentioned document", "attached file"
""",

    "find_cross_memory": """
🔍 find_cross_memory - Cross-Memory Search

Features:
- Multi-dimensional related memory search (semantic/temporal/reference/entity/keyword)
- Support for basic and advanced modes
- Fine-tunable search options
- Recency boosting and metadata integration

Usage:
# Basic search
m_memory("find_cross_memory", [memory_key])

# Advanced search
m_memory("find_cross_memory", [memory_key], {
    "advanced": True,
    "search_options": {
        "time_window_hours": 24,
        "semantic_threshold": 0.7,
        "include_types": ["conversation", "document"],
        "max_results": 20,
        "boost_recent": True,
        "entity_types": ["PERSON", "ORG"],
        "include_metadata": True
    }
})

Examples:
# Basic cross-memory search
m_memory("find_cross_memory", ["json:memory:document:2024/01/15/report.pdf"])

# Advanced search - within 48 hours, high semantic similarity
m_memory("find_cross_memory", ["stream:memory:conversation:2024/01/15/10/30/00"], {
    "advanced": True,
    "search_options": {
        "time_window_hours": 48,
        "semantic_threshold": 0.8,
        "boost_recent": True
    }
})

Return Value:
- related_conversations: List of related conversations
- related_documents: List of related documents
- temporal_neighbors: Temporally proximate memories
- shared_entities: Shared entities
- metadata: Search metadata
""",

    "analyze_memory_graph": """
📊 analyze_memory_graph - Memory Graph Analysis

Features:
- Analyze entire memory relationship network
- Identify hub nodes (centrality-based)
- Detect isolated nodes
- Analyze dense clusters and extract topics
- Temporal pattern analysis
- Auto-generate improvement recommendations

Usage:
m_memory("analyze_memory_graph", [])

Examples:
# Analyze entire memory graph
result = m_memory("analyze_memory_graph", [])

Return Value:
- total_nodes: Total number of nodes
- total_edges: Total number of connections
- connected_components: Number of connected components
- avg_clustering_coefficient: Average clustering coefficient
- hub_nodes: List of hub nodes (with centrality scores)
- isolated_nodes: List of isolated nodes
- dense_clusters: Dense cluster information
- temporal_patterns: Temporal patterns (peak times, active days)
- recommendations: AI improvement recommendations
- analysis_timestamp: Analysis timestamp

Usage Example:
print(f"Total {result['total_nodes']} memories, {result['total_edges']} connections")
print("Recommendations:")
for rec in result['recommendations']:
    print(f"- {rec}")
""",

    "suggest_connections": """
🤖 suggest_connections - AI Connection Suggestions

Features:
- Connection suggestions based on semantic similarity
- Connection suggestions based on temporal proximity
- Connection suggestions based on shared entities
- Connection suggestions based on topic keywords
- Suggest missing reference relationships
- Deduplication and confidence-based sorting

Usage:
m_memory("suggest_connections", [])

Examples:
# Get AI connection suggestions
suggestions = m_memory("suggest_connections", [])

for suggestion in suggestions:
    print(f"Suggestion: {suggestion['reasoning']}")
    print(f"Confidence: {suggestion['confidence']:.2f}")
    print(f"Benefit: {suggestion['potential_benefit']}")
    print()

Return Value (per suggestion):
- source_id: Source memory ID
- target_id: Target memory ID
- suggested_type: Suggested connection type (semantic/temporal/entity/keyword/reference)
- confidence: Confidence score (0.0-1.0)
- reasoning: Reasoning for suggestion
- potential_benefit: Expected benefit
- metadata: Additional metadata

Connection Types:
- semantic: Semantic similarity (embedding-based)
- temporal: Temporal proximity (creation time-based)
- entity: Shared entities (people, organizations, places)
- keyword: Common topic keywords
- reference: Missing reference relationships
""",

    "get_memory_insights": """
🔍 get_memory_insights - Individual Memory Insights

Features:
- Analyze importance and role of specific memory
- Calculate centrality scores (network position)
- Analyze by connection type
- Cluster membership information
- Provide importance indicators
- Personalized improvement suggestions

Usage:
m_memory("get_memory_insights", [memory_key])

Examples:
# Analyze specific memory insights
insights = m_memory("get_memory_insights", ["json:memory:document:2024/01/15/report.pdf"])

print(f"Centrality score: {insights['centrality_score']:.3f}")
print(f"Connection count: {insights['connection_count']}")
print(f"Connection types: {insights['connection_types']}")

if insights['cluster_membership']:
    cluster = insights['cluster_membership']
    print(f"Cluster: {cluster['topic']} ({cluster['size']} nodes)")

for suggestion in insights['suggestions']:
    print(f"Suggestion: {suggestion}")

Return Value:
- memory_id: Target memory ID for analysis
- centrality_score: Centrality score (0.0-1.0)
- connection_count: Direct connection count
- connection_types: Count by connection type
- cluster_membership: Cluster membership information
- importance_indicators: List of importance indicators
- related_memories: Directly connected memories
- suggestions: Personalized improvement suggestions

Importance Indicators:
- "High centrality - key information": Important position in network
- "Multiple connections - hub role": Connected to many memories
- "Cluster center": Core of topic-based group

Improvement Suggestions:
- How to connect isolated memories
- Clustering improvement methods
- Temporal connection opportunities
""",
}


def get_help_message(topic: Optional[str] = None, subtopic: Optional[str] = None) -> str:
    """Return help message.
    
    Args:
        topic: Help topic
        subtopic: Detailed subtopic
        
    Returns:
        Help message
    """
    if not topic:
        return HELP_MESSAGES["overview"]
    
    # Combine if subtopic exists
    if subtopic:
        key = f"{topic}_{subtopic}" if f"{topic}_{subtopic}" in HELP_MESSAGES else subtopic
    else:
        key = topic
    
    return HELP_MESSAGES.get(key, f"Help not found: {topic}")


def generate_example(tool: str, action: str, **kwargs) -> str:
    """Generate executable example code.
    
    Args:
        tool: Tool name
        action: Action name
        **kwargs: Additional parameters
        
    Returns:
        Example code
    """
    examples = {
        # Basic m_memory functions
        ("m_memory", "save"): 'm_memory("save", ["category"], "content")',
        ("m_memory", "search", "keyword"): 'm_memory("search", [], "search_term")',
        ("m_memory", "search", "time_range"): '''m_memory("search", [], None, {
    "type": "time_range",
    "filters": {"date": "2025-05-28"}
})''',
        
        # LRMM advanced features
        ("m_memory", "link_conversation"): 'm_memory("link_conversation", ["stream:memory:conversation:2024/01/15/10/30/00"])',
        ("m_memory", "find_cross_memory"): 'm_memory("find_cross_memory", ["json:memory:document:2024/01/15/report.pdf"])',
        ("m_memory", "find_cross_memory", "advanced"): '''m_memory("find_cross_memory", ["memory_key"], {
    "advanced": True,
    "search_options": {
        "time_window_hours": 24,
        "semantic_threshold": 0.7,
        "boost_recent": True
    }
})''',
        ("m_memory", "analyze_memory_graph"): 'm_memory("analyze_memory_graph", [])',
        ("m_memory", "suggest_connections"): 'm_memory("suggest_connections", [])',
        ("m_memory", "get_memory_insights"): 'm_memory("get_memory_insights", ["memory_key"])',
        
        # Existing functions
        ("m_consolidate", "path"): 'm_consolidate("path", ["2025", "05"])',
        ("m_lifecycle", "evaluate"): 'm_lifecycle("evaluate", ["path", "to", "memory"])',
        ("m_assistant", "save"): 'm_assistant("save project meeting content")',
    }
    
    key = (tool, action)
    if "type" in kwargs:
        key = (tool, action, kwargs["type"])
    
    return examples.get(key, f"# No example available for {tool} {action}")


def suggest_fix(error_message: str, context: Dict[str, Any]) -> str:
    """Analyze error message and suggest fixes.
    
    Args:
        error_message: Error message
        context: Execution context
        
    Returns:
        Fix suggestion
    """
    suggestions = {
        "Time range search requires time filters": """
Time range search requires time filters.

Use one of the following:
- filters: {"date": "2025-05-28"}  # Specific date
- filters: {"from": "2025-05-01", "to": "2025-05-31"}  # Date range
- filters: {"from": "2025-05-28T00:00:00", "to": "2025-05-28T23:59:59"}  # With time
""",
        "Query string is required": """
Keyword search requires a search term.

Enter search term in content parameter:
m_memory("search", [], "search_term")
""",
        "Paths are required": """
Paths are required.

Provide path list in paths parameter:
m_memory("get", ["category", "subcategory"])
""",
        "CrossMemoryBridge not initialized": """
CrossMemoryBridge is not initialized.

To use relationship features, initialize the memory engine with enable_relationships=True.
This is required for the following LRMM advanced features:
- link_conversation (automatic conversation-document linking)
- find_cross_memory (cross-memory search)
""",
        "MemoryAnalytics not initialized": """
MemoryAnalytics is not initialized.

To use analytics features, initialize the memory engine with enable_relationships=True.
This is required for the following LRMM advanced features:
- analyze_memory_graph (memory graph analysis)
- suggest_connections (AI connection suggestions)
- get_memory_insights (individual memory insights)
""",
        "Conversation ID required": """
Conversation ID is required.

The link_conversation action requires a conversation key:
m_memory("link_conversation", ["stream:memory:conversation:2024/01/15/10/30/00"])
""",
        "Memory key required": """
Memory key is required.

Provide memory key in one of these formats:
- Conversation: "stream:memory:conversation:2024/01/15/10/30/00"
- Document: "json:memory:document:2024/01/15/report.pdf"

Examples:
m_memory("find_cross_memory", ["json:memory:document:2024/01/15/report.pdf"])
m_memory("get_memory_insights", ["stream:memory:conversation:2024/01/15/10/30/00"])
""",
    }
    
    for key, suggestion in suggestions.items():
        if key in error_message:
            return suggestion
    
    # General help for LRMM features
    if any(action in error_message for action in ["link_conversation", "find_cross_memory", "analyze_memory_graph", "suggest_connections", "get_memory_insights"]):
        return """
LRMM Advanced Features Usage:

🔗 Conversation-document linking: m_memory("link_conversation", [conversation_id])
🔍 Cross-memory search: m_memory("find_cross_memory", [memory_key])
📊 Graph analysis: m_memory("analyze_memory_graph", [])
🤖 Connection suggestions: m_memory("suggest_connections", [])
🔍 Memory insights: m_memory("get_memory_insights", [memory_key])

For detailed help: m_memory("help", ["feature_name"])
"""
    
    return "An error occurred. Please refer to help: m_memory(\"help\", [])"