# MCP Flow and Swimlane Diagrams

This document contains Mermaid diagrams that visualize the flow and architecture of the Medical Report Processing application with MCP.

## System Architecture Flow Diagram

```mermaid
flowchart LR
    User([User]):::userClass
    WebApp[Web Application\n(Flask)]:::webClass
    MCPClient[MCP Client]:::mcpClass
    MCPServer[MCP Server]:::mcpClass
    Tools[MCP Tools]:::mcpClass
    Storage[(Results Storage)]:::storageClass
    
    User -- 1. Upload Medical Report --> WebApp
    WebApp -- 2. Detect Tool --> WebApp
    WebApp -- 3. Connect & Call Tool --> MCPClient
    MCPClient -- 4. Initialize Session --> MCPServer
    MCPClient -- 5. Call Tool --> MCPServer
    MCPServer -- 6. Execute --> Tools
    Tools -- 7. Return Results --> MCPServer
    MCPServer -- 8. Return Results --> MCPClient
    MCPClient -- 9. Return Results --> WebApp
    WebApp -- 10. Save Results --> Storage
    WebApp -- 11. Display "Process Completed" --> User
    User -- 12. Click "View Results" --> WebApp
    WebApp -- 13. Fetch Results --> Storage
    WebApp -- 14. Display Formatted Results --> User
    
    classDef userClass fill:#f9f,stroke:#333,stroke-width:2px;
    classDef webClass fill:#bbf,stroke:#33f,stroke-width:2px;
    classDef mcpClass fill:#bfb,stroke:#3f3,stroke-width:2px;
    classDef storageClass fill:#fbb,stroke:#f33,stroke-width:2px;
```

## MCP Communication Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant WebApp as Web Application
    participant MCPClient as MCP Client
    participant MCPServer as MCP Server
    participant Tools as MCP Tools
    participant Storage as Results Storage
    
    User->>WebApp: Upload Medical Report
    WebApp->>WebApp: Detect Tool Type
    
    WebApp->>MCPClient: Create Client Instance
    activate MCPClient
    
    MCPClient->>MCPServer: Connect (stdio transport)
    activate MCPServer
    MCPServer-->>MCPClient: Connection Established
    
    MCPClient->>MCPServer: Initialize Session
    MCPServer-->>MCPClient: Session Initialized
    
    MCPClient->>MCPServer: Call Tool (process_medical_report/summarize_medical_report)
    MCPServer->>Tools: Execute Tool
    activate Tools
    
    Tools->>Tools: Extract Text from Report
    Tools->>Tools: Process/Summarize Text
    Tools->>Tools: Generate Structured Results
    
    Tools-->>MCPServer: Return Results
    deactivate Tools
    
    MCPServer-->>MCPClient: Return Results
    deactivate MCPServer
    
    MCPClient-->>WebApp: Return Results
    deactivate MCPClient
    
    WebApp->>Storage: Save Results to JSON
    WebApp-->>User: Display "Process Completed" Message
    
    User->>WebApp: Click "View Results"
    WebApp->>Storage: Fetch Results
    Storage-->>WebApp: Return Results
    WebApp-->>User: Display Formatted Results
```

## MCP Swimlane Diagram

```mermaid
graph TD
    %% Define the swimlanes
    subgraph User ["User"]
        U1["Upload Medical Report"]:::userAction
        U2["View Completion Message"]:::userAction
        U3["Click View Results"]:::userAction
        U4["View Detailed Results"]:::userAction
    end
    
    subgraph WebApp ["Web Application (Flask)"]
        W1["Receive Upload"]:::webAction
        W2["Save File"]:::webAction
        W3["Detect Tool Type"]:::webAction
        W4["Create MCP Client"]:::webAction
        W7["Save Results to JSON"]:::webAction
        W8["Return Simple Message"]:::webAction
        W9["Fetch Results"]:::webAction
        W10["Render Results View"]:::webAction
    end
    
    subgraph MCP ["MCP Client & Server"]
        M1["Connect to Server"]:::mcpAction
        M2["Initialize Session"]:::mcpAction
        M3["Call Tool"]:::mcpAction
        M4["Execute Tool"]:::mcpAction
        M5["Return Results"]:::mcpAction
    end
    
    subgraph Tools ["MCP Tools"]
        T1["Process/Summarize Report"]:::toolAction
        T2["Extract Text"]:::toolAction
        T3["Generate Structured Data"]:::toolAction
    end
    
    subgraph Storage ["Results Storage"]
        S1["Save Results"]:::storageAction
        S2["Retrieve Results"]:::storageAction
    end
    
    %% Define the flow
    U1 --> W1
    W1 --> W2
    W2 --> W3
    W3 --> W4
    W4 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> T1
    T1 --> T2
    T2 --> T3
    T3 --> M5
    M5 --> W7
    W7 --> S1
    W7 --> W8
    W8 --> U2
    U3 --> W9
    W9 --> S2
    S2 --> W10
    W10 --> U4
    
    %% Styling
    classDef userAction fill:#f9f,stroke:#333,stroke-width:1px;
    classDef webAction fill:#bbf,stroke:#33f,stroke-width:1px;
    classDef mcpAction fill:#bfb,stroke:#3f3,stroke-width:1px;
    classDef toolAction fill:#fbf,stroke:#f3f,stroke-width:1px;
    classDef storageAction fill:#fbb,stroke:#f33,stroke-width:1px;
```

## MCP Tool Registration and Execution Flow

```mermaid
flowchart LR
    %% Server Initialization
    SI1["Create FastMCP Instance"]:::init --> SI2["Configure Host/Port"]:::init
    SI2 --> SI3["Register Tools"]:::init
    
    %% Tool Registration
    SI3 --> TR1["@mcp.tool Decorator"]:::reg
    TR1 --> TR2["Define Tool Function"]:::reg
    TR2 --> TR3["Implement Tool Logic"]:::reg
    
    %% Server Startup
    TR3 --> SS1["Parse Arguments"]:::start
    SS1 --> SS2["Select Transport (stdio)"]:::start
    SS2 --> SS3["Run Server"]:::start
    
    %% Tool Execution
    SS3 --> TE1["Receive Tool Call"]:::exec
    TE1 --> TE2["Parse Arguments"]:::exec
    TE2 --> TE3["Execute Tool Function"]:::exec
    TE3 --> TE4["Return Results"]:::exec
    
    classDef init fill:#bbf,stroke:#33f,stroke-width:1px;
    classDef reg fill:#bfb,stroke:#3f3,stroke-width:1px;
    classDef start fill:#fbf,stroke:#f3f,stroke-width:1px;
    classDef exec fill:#fbb,stroke:#f33,stroke-width:1px;
```

## MCP Client-Server Communication

```mermaid
sequenceDiagram
    participant WebApp as Web Application
    participant MCPClient as MCP Client
    participant Transport as stdio Transport
    participant MCPServer as MCP Server
    participant Tool as MCP Tool
    
    WebApp->>MCPClient: Create Client
    
    MCPClient->>MCPClient: Create AsyncExitStack
    
    MCPClient->>Transport: Create stdio Transport
    activate Transport
    Note over MCPClient,Transport: StdioServerParameters with<br>python mcp_chat.py --transport=stdio
    
    Transport->>MCPServer: Start Server Process
    activate MCPServer
    MCPServer->>MCPServer: Initialize FastMCP
    MCPServer->>MCPServer: Register Tools
    MCPServer->>MCPServer: Start stdio Transport
    
    Transport-->>MCPClient: Return stdio, write
    
    MCPClient->>MCPClient: Create ClientSession
    MCPClient->>MCPServer: Initialize Session
    MCPServer-->>MCPClient: Session Initialized
    
    WebApp->>MCPClient: Call Tool
    MCPClient->>MCPServer: Call Tool Request
    MCPServer->>Tool: Execute Tool
    activate Tool
    
    Tool->>Tool: Process File
    Tool-->>MCPServer: Return Results
    deactivate Tool
    
    MCPServer-->>MCPClient: Return Results
    MCPClient-->>WebApp: Return Results
    
    WebApp->>MCPClient: Close Connection
    MCPClient->>Transport: Close Transport
    deactivate Transport
    Transport->>MCPServer: Terminate Server
    deactivate MCPServer
```

## MCP Tool Detection and Selection

```mermaid
flowchart TD
    Start(["Start"]):::start --> HasFile{"Has File?"}:::decision
    
    HasFile -->|"Yes, No Message"| DefaultProcess["Use process_medical_report"]:::action
    HasFile -->|"Yes, With Message"| CheckKeywords["Check Message Keywords"]:::process
    HasFile -->|"No"| CheckKeywords
    
    CheckKeywords --> CountSummarize["Count Summarize Keywords"]:::process
    CountSummarize --> CountProcess["Count Process Keywords"]:::process
    
    CountProcess --> CompareKeywords{"Compare Keyword Counts"}:::decision
    
    CompareKeywords -->|"Summarize > Process"| UseSummarize["Use summarize_medical_report"]:::action
    CompareKeywords -->|"Process > 0 or Has File"| UseProcess["Use process_medical_report"]:::action
    CompareKeywords -->|"No Clear Match"| UseGeneral["Handle as General Question"]:::action
    
    DefaultProcess --> ReturnTool["Return Tool Name and Confidence"]:::action
    UseSummarize --> ReturnTool
    UseProcess --> ReturnTool
    UseGeneral --> ReturnTool
    
    ReturnTool --> End(["End"]):::end
    
    classDef start fill:#f9f,stroke:#333,stroke-width:1px;
    classDef end fill:#f9f,stroke:#333,stroke-width:1px;
    classDef decision fill:#bfb,stroke:#3f3,stroke-width:1px;
    classDef process fill:#bbf,stroke:#33f,stroke-width:1px;
    classDef action fill:#fbb,stroke:#f33,stroke-width:1px;
```
